from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from config import llm
from tools import ALL_TOOLS

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = SystemMessage(content =(
    "You are a youtube Learning Assistant. You Help users understand"
    "Youtube video content by fetching transcripts and answering"
    "questions using retrieval-agumented search over the transcript.\n\n"
    "Rules :\n"
    "- If no video has been loaded yet and the user asks about video"
    "content, call load_youtube_video first (ask for a URL if you don't )"
    "have one).\n"
    "- Use search_video_content to find relavant transcript excerpts"
    "before answering content questions - don't guess.\n"
    "- Base your answers only on retrived transcript content. If the"
    "content doesn't cover something, say so honestly instead of"
    "making it up.\n"
    "- Be concise and clear, like a good tutor"
))

def agent_node(state: AgentState) -> dict:
    """
    The 'Brain' node.Sends the full conversation so 
    far to the llm and gets back either:
        a. A normal text response(the agent is done), or
        b. A response containing tool_calls (the agent wants to act)
    """

    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(ALL_TOOLS)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END : END}
)

graph_builder.add_edge("tools","agent")

agent_graph = graph_builder.compile()

if __name__ == "__main__":
    result = agent_graph.invoke({
        "message": [{"role": "user", "content" : "Hi, What can you help me with ?"}]
    })
    print(result["messages"][-1].content)
