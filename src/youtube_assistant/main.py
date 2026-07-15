from src.youtube_assistant.agent.agent_graph import THIS_DIR
import asyncio
import os, uuid

from rich.console import Console
from rich.markdown import Markdown
from youtube_assistant.agent.agent_graph import build_agent_graph
from settings import THREAD_ID_FILE


console = Console()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THREAD_ID_PATH = os.path.join(THIS_DIR, THREAD_ID_FILE)

def _load_or_create_thread_it() -> str:
    """
    Reads the saed thread_id from disk if one exists (resuming last
    conversation); otherwise creates a new one and saves it.
    """
    if os.path.exists(THREAD_ID_PATH):
        with open(THREAD_ID_PATH, "r") as f:
            saved = f.read().strip()
            if saved:
                return saved
    return _create_new_thread_id()

def _create_new_thread_id() -> str:
    new_id = str(uuid.uuid4())
    with open(THREAD_ID_PATH, "w") as f:
        f.write(new_id)
    return new_id

async def main():
    console.print(
        "[bold cyan]YouTube Learning Assistant[/bold cyan]"
        "(Part 1: Single Agent + RAG, no MCP Agent)\n"
        "Paste a YouTube URL to load it, then ask questions.\n"
        "Type 'exit' to quit.\n"
    )

    thread_id = _load_or_create_thread_it()

    async with build_agent_graph() as agent_graph:
        config = {"configurable" : {"thread_id": thread_id}}
        existing_state = await agent_graph.aget_state(config)
        resumed = bool(existing_state.values.get("messages"))
        console.print(
            "[dim]Connected. Supervisor + 3 specialists ready "
            "(loader / qa / summary).[/dim]\n"
        )
        console.print(f"[dim]Thread: {thread_id}" + (" (resumed)" if resumed else " (new)") + "[/dim]\n")
        console.print(
            "Paste a YouTube URL to load it, then ask questions or "
            "request a summary.\n"
            "Type 'exit' to quit.\n"
        )

        while True: 
            user_input = console.input("[bold green]You:[/bold green]").strip()
            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]GoodBye![/dim]")
                break
            if not user_input:
                continue
            if user_input.lower() == "/new":
                thread_id = _create_new_thread_id()
                config = {"configurable": {"thread_id": thread_id}}
                console.print(f"[dim]Started a new conversation. Thread : {thread_id}[/dim]\n")
                continue

            try:
                result = await agent_graph.ainvoke(
                    {"messages" [{"role": "user", "content" : user_input}]},
                    config
                )
            except Exception as e:
                console.print(f"[bold red]Error: [/bold red] {e}")
                continue

            final_message = result["messages"][-1]
            console.print("[bold magenta]Assistant: [/bold magenta]")
            console.print(Markdown(final_message.content))
            console.print()

if __name__ =="__main__":
    asyncio.run(main())        