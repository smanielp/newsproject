from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
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
        tool_message = ToolMessage(content=news_results, name="news_tool")
        return {"messages": [tool_message]}

    def call_web_search_tool(self, state: GraphState):
        last_message = state['messages'][-1].content
        search_results = web_search(last_message)
        tool_message = ToolMessage(content=search_results, name="web_search_tool")
        return {"messages": [tool_message]}

    def generate_response(self, state: GraphState):
        response = self.llm.invoke(state['messages'])
        return {"messages": [response]}

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
