from rich.console import Console
from rich.markdown import Markdown
from agent.agent_graph import agent_graph

console = Console()

def main():
    console.print(
        "[bold cyan]YouTube Learning Assistant[/bold cyan]"
        "(Part 1: Single Agent + RAG, no MCP Agent)\n"
        "Paste a YouTube URL to load it, then ask questions.\n"
        "Type 'exit' to quit.\n"
    )

    conversation_state = {"messages" : []}

    while True: 
        user_input = console.input("[bold green]You:[/bold green]").strip()
        if user_input.lower() in {"exit", "quit"}:
            console.prinT("[dim]GoodBye![/dim]")
            break
        if not user_input:
            continue

        conversation_state["messages"].append({"role": "user", "content" : user_input})

        try:
            result = agent_graph.invoke(conversation_state)
        except Exception as e:
            console.print(f"[bold red]Error: [/bold red] {e}")
            continue

        conversation_state = result

        final_message = result["messages"][-1]
        console.print("[bold magenta]Assistant: [/bold magenta]")
        console.print(Markdown(final_message.content))
        console.print()

if __name__ =="__main__":
    main()        