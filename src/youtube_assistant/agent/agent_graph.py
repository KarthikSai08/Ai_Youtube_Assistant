
from typing import Annotated
from typing_extensions import TypedDict
from pathlib import Path
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
# pyrefly: ignore [missing-import]
from langchain_mcp_adapters.client import MultiServerMCPClient
# pyrefly: ignore [missing-import]
from langchain_mcp_adapters.tools import load_mcp_tools

from youtube_assistant.config import llm

load_dotenv()
THIS_DIR = Path(__file__).parent

mcp_client = MultiServerMCPClient(
    {
        "youtube_rag" : {
            "command" : "python",
            "args" : [str(THIS_DIR.parent / "mcp" / "mcp_server.py")],
            "transport" : "stdio"
        }
    }
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

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

def _build_graph_from_tools(mcp_tools: list):
    """
    Pure Graph-construction logic, separated from session managment so
    it's easy to read on its own. 
    """

    llm_with_tools = llm.bind_tools(mcp_tools)
    async def agent_node(state: AgentState) -> dict:
        """
        The 'Brain' node.Sends the full conversation so 
        far to the llm and gets back either:
            a. A normal text response(the agent is done), or
            b. A response containing tool_calls (the agent wants to act)
        """
        response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(mcp_tools)

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

    return graph_builder.compile()

@asynccontextmanager
async def build_agent_graph():
    """
     Async context manager: opens ONE persistent MCP session for the
    "youtube_rag" server, loads its tools from that session, builds the
    graph, and yields it. The session (and the underlying subprocess)
    stays alive for as long as the `async with` block is open — which in
    main.py is the entire lifetime of the chat loop.

    WHY A CONTEXT MANAGER (vs just returning the graph):
    We need to guarantee the subprocess gets cleanly shut down when the
    program exits (or crashes), not left running as an orphan process.
    `async with` guarantees the cleanup code runs no matter how the block
    exits — same reason you use `with open(...)` for files.
    """
    async with mcp_client.session("youtube_rag") as session:
        mcp_tools = await load_mcp_tools(session)
        graph = _build_graph_from_tools(mcp_tools)
        yield graph

if __name__ == "__main__":
    import asyncio

    async def smoke_test():
        async with build_agent_graph() as graph:
            result = await graph.ainvoke({
                "messages" : [{"role": "user", "content" : "Hi, what can you help me with?"}]
            })
            print(result["messages"][-1].content)
    
    asyncio.run(smoke_test())

