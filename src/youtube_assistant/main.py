import asyncio
from rich.console import Console
from rich.markdown import Markdown
from youtube_assistant.agent.agent_graph import build_agent_graph

console = Console()

async def main():
    console.print(
        "[bold cyan]YouTube Learning Assistant[/bold cyan]"
        "(Part 1: Single Agent + RAG, no MCP Agent)\n"
        "Paste a YouTube URL to load it, then ask questions.\n"
        "Type 'exit' to quit.\n"
    )
    async with build_agent_graph() as agent_graph:
        console.print(
            "[dim]Connected. Tools are now being served over MCP.[/dim]\n"
            "Paste a YouTube URL to load it, then ask questions.\n"
            "Type 'exit' to quit.\n"
        )

        conversation_state = {"messages" : []}

        while True: 
            user_input = console.input("[bold green]You:[/bold green]").strip()
            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]GoodBye![/dim]")
                break
            if not user_input:
                continue

            conversation_state["messages"].append({"role": "user", "content" : user_input})

            try:
                result = await agent_graph.ainvoke(conversation_state)
            except Exception as e:
                console.print(f"[bold red]Error: [/bold red] {e}")
                continue

            conversation_state = result

            final_message = result["messages"][-1]
            console.print("[bold magenta]Assistant: [/bold magenta]")
            console.print(Markdown(final_message.content))
            console.print()

if __name__ =="__main__":
    asyncio.run(main())        