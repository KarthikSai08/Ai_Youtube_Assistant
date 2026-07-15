from concurrent.futures import thread
from typing import Annotated
from typing_extensions import TypedDict
from pathlib import Path
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, ToolMessage

# pyrefly: ignore [missing-import]
from langchain_mcp_adapters.client import MultiServerMCPClient
# pyrefly: ignore [missing-import]
from langchain_mcp_adapters.tools import load_mcp_tools
from youtube_assistant.config import llm
from youtube_assistant.settings import CHECKPOINT_DB_PATH


load_dotenv()
THIS_DIR = Path(__file__).parent

mcp_client = MultiServerMCPClient(
    {
        "youtube_rag" : {
            "command" : "python",
            "args" : [str(THIS_DIR.parent / "mcp-tools" / "mcp_server.py")],
            "transport" : "stdio"
        }
    }
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def _make_handoff_tool(agent_name: str, tool_name: str, description: str):
    @tool(tool_name, description= description)
    def handoff(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        confirmation = ToolMessage(
            content = f"Transferring to {agent_name}.",
            name = tool_name,
            tool_call_id = tool_call_id
        )
        return Command(goto = agent_name, update = {"messages" : [confirmation]})
    return handoff

transfer_to_loader = _make_handoff_tool(
    "loader_agent", "transfer_to_loader",
    "Use when the user gives a YouTube URL/video to load, or asks to "
    "analyze/work with a video and none is loaded yet, or wants to "
    "switch to a different video.",
)
transfer_to_qa = _make_handoff_tool(
    "qa_agent", "transfer_to_qa",
    "Use when the user asks a SPECIFIC question about the currently "
    "loaded video's content (e.g. 'what does it say about X', 'does the "
    "video mention Y').",
)
transfer_to_summary = _make_handoff_tool(
    "summary_agent", "transfer_to_summary",
    "Use when the user wants an OVERVIEW, SUMMARY or Key takeawways of "
    "the currently loaded video as a whole - not an answer to a "
    "specific question."
)
transfer_to_timestamp = _make_handoff_tool(
    "timestamp_agent", "transfer_to_timestamp",
    "Use when the user asks WHEN something happens in the video — "
    "e.g. 'when does it talk about X', 'what time does Y happen', "
    "'find the part about Z'. This is about locating a MOMENT, not "
    "answering a content question or summarizing.",
)
transfer_to_quiz = _make_handoff_tool(
    "quiz_agent", "transfer_to_quiz",
    "Use when the user wants to be QUIZZED or TESTED on the video's "
    "content — e.g. 'quiz me', 'make a quiz', 'test my understanding', "
    "'give me some practice questions'.",
)

HANDOFF_TOOLS = [transfer_to_loader, transfer_to_qa, transfer_to_summary, 
                 transfer_to_timestamp, transfer_to_quiz]

SUPERVISOR_PROMPT = SystemMessage(content=(
    "You are the supervisor of a YouTube Learning Assistant team. You do "
    "NOT answer video-content questions yourself — you ONLY decide which "
    "specialist should handle the user's request, by calling exactly one "
    "of your transfer tools:\n"
    "- transfer_to_loader: to load/index a video\n"
    "- transfer_to_qa: to answer a specific question about loaded video content\n"
    "- transfer_to_summary: to summarize/give an overview of the loaded video\n"
    "- transfer_to_timestamp: to find WHEN something happens in the video\n"
    "- transfer_to_quiz: to quiz/test the user on the video's content\n\n"
    "If the user is just greeting you, asking what you can do, or making "
    "small talk unrelated to a video, respond directly in plain text — "
    "do NOT call a transfer tool for that.\n"
    "If a specialist reports back that its part is done and the "
    "original request had another part left (e.g. 'load this video and "
    "quiz me on it' — after loading, quizzing is still pending), route "
    "to the next appropriate specialist instead of ending.\n"
    "You have access to the full conversation history, including earlier "
    "videos loaded and questions asked in previous sessions — use that "
    "context naturally when relevant (e.g. if the user says 'now quiz me "
    "on that' referring to something discussed earlier)."
))

LOADER_PROMPT = SystemMessage(content=(
    "You are the Loader specialist. Your only job is to load and index "
    "the YouTube video the user (or supervisor) referenced, using your "
    "tools. Once loading succeeds (or fails with a clear reason), report "
    "the outcome in ONE concise sentence. Do not attempt to answer "
    "content questions — that's not your job."
))

QA_PROMPT = SystemMessage(content=(
    "You are the Q&A specialist. Use search_video_content to find "
    "transcript excerpts relevant to the user's specific question, then "
    "answer using ONLY that retrieved content. If the content doesn't "
    "cover the question, say so honestly rather than guessing."
))

SUMMARY_PROMPT = SystemMessage(content=(
    "You are the Summary specialist. Use get_transcript_overview to "
    "retrieve the video's transcript, then produce a clear, well-"
    "organized summary or key-takeaways list, as appropriate to what the "
    "user asked for. Base it only on the retrieved transcript text."
))

TIMESTAMP_PROMPT = SystemMessage(content=(
    "You are the Timestamp specialist. Use find_timestamp to locate the "
    "moment(s) in the video relevant to what the user is asking about. "
    "Report the timestamp(s) clearly (e.g. 'Around 3:45, the video "
    "discusses...') along with a brief description of what happens "
    "there, based on the retrieved snippet. If nothing relevant is "
    "found, say so honestly."
))

QUIZ_PROMPT = SystemMessage(content=(
    "You are the Quiz specialist. Use get_transcript_overview to "
    "retrieve the video's content, then write a short quiz of 4-5 "
    "multiple-choice questions (A/B/C/D) testing understanding of the "
    "video's key points. Base every question strictly on the retrieved "
    "content — do not invent facts not in the transcript. After all the "
    "questions, include an 'Answer key' section listing the correct "
    "letter for each question number."
))


async def _run_specialist_loop(llm_with_tools, tool_by_name: dict, system_prompt : SystemMessage,
                                messages: list, max_iterations : int = 4) -> list:
    """
    A small, explicit ReAct loop for ONE specialist: call the LLM, and if
    it requests tool calls, execute them directly and feed the results
    back, repeating until the LLM answers in plain text (or we hit
    max_iterations, as a safety valve against infinite loops).

    Returns the list of NEW messages produced (not including the input
    `messages`), so the caller can append just the delta to graph state.
    """
    working_messages = list(messages)
    new_messages = []

    for _ in range(max_iterations):
        response = await llm_with_tools.ainvoke([system_prompt] + working_messages)
        working_messages.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            tool_fn = tool_by_name[call["name"]]
            result = await tool_fn.ainvoke(call["args"])
            tool_message = ToolMessage(
                content = str(result), name = call["name"], tool_call_id = call["id"]
            )
            working_messages.append(tool_message)
            new_messages.append(tool_message)

    return new_messages

def _build_graph_from_tools(mcp_tools: list):
    tools_by_name = {t.name: t for t in mcp_tools}

    loader_tools = [tools_by_name["load_youtube_video"], tools_by_name["get_video_id"]]
    qa_tools = [tools_by_name["search_video_content"]]
    summary_tools = [tools_by_name["get_transcript_overview"]]
    timestamp_tools = [tools_by_name["find_timestamp"]]
    quiz_tools = [tools_by_name["get_transcript_overview"]]
    
    loader_llm = llm.bind_tools(loader_tools)
    qa_llm = llm.bind_tools(qa_tools)
    summary_llm = llm.bind_tools(summary_tools)
    timestamp_llm = llm.bind_tools(timestamp_tools)
    quiz_llm = llm.bind_tools(quiz_tools)
    supervisor_llm = llm.bind_tools(HANDOFF_TOOLS)

    async def supervisor_node(state: AgentState):
        response = await supervisor_llm.ainvoke([SUPERVISOR_PROMPT] + state["messages"])
        return {"messages" : [response]}

    def supervisor_router(state:AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "supervisor_tools"
        return END

    supervisor_tools_node = ToolNode(HANDOFF_TOOLS)

    async def loader_agent(state: AgentState) -> Command:
        loader_tools_by_name = {t.name: t for t in loader_tools}
        new_msgs = await _run_specialist_loop(loader_llm, loader_tools_by_name, LOADER_PROMPT, state["messages"])
        return Command(goto ="supervisor", update = {"messages" : new_msgs})

    async def qa_agent(state : AgentState) -> Command:
        qa_tools_by_name = {t.name : t for t in qa_tools}
        new_msgs = await _run_specialist_loop(qa_llm,qa_tools_by_name,QA_PROMPT, state["messages"])
        return Command(goto = END, update = {"messages":  new_msgs})

    async def summary_agent(state : AgentState) -> Command:
        summary_tools_name = {t.name : t for t in summary_tools}
        new_msgs = await _run_specialist_loop(summary_llm, summary_tools_name, SUMMARY_PROMPT, state["messages"])
        return Command(goto = END, update = {"messages" : new_msgs})

    async def timestamp_agent(state: AgentState) -> Command:
        timestamp_tools_by_name = {t.name: t for t in timestamp_tools}
        new_msgs = await _run_specialist_loop(timestamp_llm, timestamp_tools_by_name, TIMESTAMP_PROMPT, state["messages"])
        return Command(goto=END, update={"messages": new_msgs})

    async def quiz_agent(state: AgentState) -> Command:
        quiz_tools_by_name = {t.name: t for t in quiz_tools}
        new_msgs = await _run_specialist_loop(quiz_llm, quiz_tools_by_name, QUIZ_PROMPT, state["messages"])
        return Command(goto=END, update={"messages": new_msgs})

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("supervisor_tools", supervisor_tools_node)
    graph_builder.add_node("loader_agent", loader_agent)
    graph_builder.add_node("qa_agent", qa_agent)
    graph_builder.add_node("summary_agent", summary_agent)
    graph_builder.add_node("timestamp_agent", timestamp_agent)
    graph_builder.add_node("quiz_agent", quiz_agent)

    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {"supervisor_tools": "supervisor_tools", END : END}
    )

    return graph_builder.compile()

@asynccontextmanager
async def build_agent_graph():
    async with mcp_client.session("youtube_rag") as session:
        mcp_tools = await load_mcp_tools(session)
        checkpoint_path = os.path.join(THIS_DIR, CHECKPOINT_DB_PATH)
        async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
            graph = _build_graph_from_tools(mcp_tools, checkpointer)
            yield graph

if __name__ == "__main__":
    import asyncio
    import uuid

    async def smoke_test():
        async with build_agent_graph() as graph:
            thread_id = str(uuid.uuid4())
            config = {"configurable" : {"thread_id": thread_id}}
            result = await graph.ainvoke({
                "messages" : [{"role": "user", "content" : "Hi, what can you help me with?"}]},
                config
                )
            print(result["messages"][-1].content)
            
    asyncio.run(smoke_test())