import argparse
from collections.abc import Iterator
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from .ai_agent import AIAgent
from .config import AgentConfig, get_available_models, load_config
from .conversation_manager import ConversationManager


def create_ai_agent_from_config(config: AgentConfig) -> AIAgent:
    """Create an AIAgent instance from configuration dictionary."""
    return AIAgent(
        name=config.name,
        model=config.model,
        system_prompt=config.system_prompt,
        temperature=config.temperature or 0.8,
        ctx_size=config.ctx_size or 2048,
    )


def create_ai_agent_from_input(console: Console, agent_number: int) -> AIAgent:
    console.print(f"=== Creating AI Agent {agent_number} ===", style="bold cyan")

    available_models = get_available_models()
    console.print("\nAvailable Models:", style="bold")
    for model in available_models:
        console.print(Text("• " + model))
    console.print("")

    while True:
        model_completer = WordCompleter(available_models, ignore_case=True)
        model_name = (
            prompt(
                f"Enter model name (default: {available_models[0]}): ",
                completer=model_completer,
                complete_while_typing=True,
            )
            or available_models[0]
        )

        if model_name in available_models:
            break

        console.print("Invalid model name!", style="bold red")

    while True:
        try:
            temperature_str: str = prompt("Enter temperature (default: 0.8): ") or "0.8"
            temperature: float = float(temperature_str)
            if not (0.0 <= temperature <= 1.0):
                raise ValueError("Temperature must be between 0.0 and 1.0")
            break
        except ValueError as e:
            console.print(f"Invalid input: {e}", style="bold red")

    while True:
        try:
            ctx_size_str: str = prompt("Enter context size (default: 2048): ") or "2048"
            ctx_size: int = int(ctx_size_str)
            if ctx_size < 0:
                raise ValueError("Context size must be a non-negative integer")
            break
        except ValueError as e:
            console.print(f"Invalid input: {e}", style="bold red")

    name = prompt(f"Enter name (default: AI {agent_number}): ") or f"AI {agent_number}"
    system_prompt = prompt(f"Enter system prompt for {name}: ")

    return AIAgent(
        name=name,
        model=model_name,
        temperature=temperature,
        ctx_size=ctx_size,
        system_prompt=system_prompt,
    )


def markdown_to_text(markdown_content: str) -> Text:
    """Convert Markdown content to a styled Text object."""
    console = Console()
    md = Markdown(markdown_content)
    segments = list(console.render(md))
    result = Text()
    for segment in segments:
        _ = result.append(segment.text, style=segment.style)

    result.rstrip()
    return result


def display_message(
    console: Console,
    agent_name: str,
    name_color: str,
    message_stream: Iterator[str],
    use_markdown: bool = False,
):
    """
    Display a message from an agent in the console.

    Args:
        console (Console): Rich console instance.
        agent_name (str): Name of the agent.
        name_color (str): Color to use for the agent name.
        message_stream (Iterator[str]): Stream of message chunks.
        use_markdown (bool, optional): Whether to use Markdown for text formatting. Defaults to False.
    """
    # Create the agent name prefix as a Text object.
    agent_prefix = Text.from_markup(f"[{name_color}]{agent_name}[/{name_color}]: ")

    content = ""
    with Live("", console=console, transient=False, refresh_per_second=10) as live:
        for chunk in message_stream:
            content += chunk
            # Create a group that holds both the agent prefix and the content.
            content_text = markdown_to_text(content) if use_markdown else Text(content)
            live.update(agent_prefix + content_text, refresh=True)


def prompt_bool(prompt_text: str, default: bool = False) -> bool:
    response = prompt(prompt_text).lower()

    if not response or response not in ["y", "yes", "n", "no"]:
        return default

    return response[0] == "y"


def main():
    parser = argparse.ArgumentParser(description="Run a conversation between AI agents")
    _ = parser.add_argument(
        "-o", "--output", type=Path, help="Path to save the conversation log to"
    )
    _ = parser.add_argument(
        "-c", "--config", type=Path, help="Path to JSON or YAML configuration file"
    )
    args = parser.parse_args()

    color1: str = "blue"
    color2: str = "green"

    console = Console()
    console.clear()

    if args.config:
        if not args.config.suffix.lower() in ['.json', '.yaml', '.yml']:
            console.print("Config file must be either JSON or YAML format", style="bold red")
            return

        # Load from config file
        config = load_config(args.config)
        agents = [
            AIAgent(
                name=agent_config.name,
                model=agent_config.model,
                temperature=agent_config.temperature,
                ctx_size=agent_config.ctx_size,
                system_prompt=agent_config.system_prompt,
            )
            for agent_config in config.agents
        ]
        settings = config.settings
        use_markdown = settings.use_markdown or False
        allow_termination = settings.allow_termination or False
        initial_message = settings.initial_message
    else:
        # Interactive agent creation
        num_agents = 0
        while num_agents < 2:
            try:
                num_agents = int(prompt("Enter number of agents (minimum 2): "))
                if num_agents < 2:
                    console.print("Minimum 2 agents required!", style="bold red")
            except ValueError:
                console.print("Please enter a valid number!", style="bold red")
        
        agents = []
        for i in range(num_agents):
            console.clear()
            agent = create_ai_agent_from_input(console, i + 1)
            agents.append(agent)
        
        use_markdown = prompt_bool(
            "Use Markdown for text formatting? (y/N): ", default=False
        )
        allow_termination = prompt_bool(
            "Allow AI agents to terminate the conversation? (y/N): ", default=False
        )
        initial_message = prompt("Enter initial message (can be empty): ") or None

        console.clear()

    # Update color handling for multiple agents
    colors = ["blue", "green", "yellow", "red", "magenta", "cyan"]
    agent_colors = {agent.name: colors[i % len(colors)] for i, agent in enumerate(agents)}

    manager = ConversationManager(
        agents=agents,
        initial_message=initial_message,
        use_markdown=use_markdown,
        allow_termination=allow_termination,
    )

    # Update the display code to use agent_colors
    console.print("=== Conversation Started ===\n", style="bold cyan")
    is_first_message = True

    try:
        for agent_name, message in manager.run_conversation():
            if not is_first_message:
                console.print("")
                console.rule()
                console.print("")

            is_first_message = False
            color = agent_colors[agent_name]
            display_message(console, agent_name, color, message, use_markdown)

    except KeyboardInterrupt:
        pass

    console.print("\n=== Conversation Ended ===\n", style="bold cyan")

    if args.output is not None:
        manager.save_conversation(args.output)
        console.print(f"\nConversation saved to {args.output}\n\n", style="bold yellow")


if __name__ == "__main__":
    main()
