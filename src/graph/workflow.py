from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.tools.news import get_news
from src.tools.search import web_search

class RouteQuery(BaseModel):
    """Route the user's query to the appropriate tool."""
    tool_name: str = Field(..., description="The name of the tool to use.", enum=["news", "web_search"])

class GraphState(TypedDict):
    messages: List[BaseMessage]
    category: Optional[str]
    tool_output: Optional[str]

class NewsGenieWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def route_query(self, state: GraphState):
        last_message = state['messages'][-1].content
        structured_llm = self.llm.with_structured_output(RouteQuery)
        routing_result = structured_llm.invoke(f"Is the following query a request for news or a general question that can be answered by a web search? Query: '{last_message}'")
        return routing_result.tool_name

    def call_news_tool(self, state: GraphState):
        category = state.get("category", "business")
        news_results = get_news(category)
        return {"tool_output": news_results}

    def call_web_search_tool(self, state: GraphState):
        last_message = state['messages'][-1].content
        search_results = web_search(last_message)
        return {"tool_output": search_results}

    def generate_response(self, state: GraphState):
        last_message = state['messages'][-1].content
        tool_output = state.get('tool_output')

        # Craft a prompt that includes the tool's output
        prompt = (
            f"Based on the following information: {tool_output}\n\n"
            f"Please provide a conversational response to the user's query: '{last_message}'"
        )

        response = self.llm.invoke(prompt)
        ai_message = AIMessage(content=response.content)

        # Append the new AI message to the conversation history
        return {"messages": state['messages'] + [ai_message]}

    def build(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("news_tool", self.call_news_tool)
        workflow.add_node("web_search_tool", self.call_web_search_tool)
        workflow.add_node("generate_response", self.generate_response)

        workflow.set_conditional_entry_point(
            self.route_query,
            {
                "news": "news_tool",
                "web_search": "web_search_tool",
            },
        )

        workflow.add_edge("news_tool", "generate_response")
        workflow.add_edge("web_search_tool", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()
