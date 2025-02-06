from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, List

from .ai_agent import AIAgent


@dataclass
class ConversationManager:
    class ConversationLogItem(TypedDict):
        agent: str
        content: str

    agents: List[AIAgent]
    initial_message: str | None = None
    use_markdown: bool = False
    allow_termination: bool = False
    _conversation_log: list[ConversationLogItem] = field(
        default_factory=list, init=False
    )
    _current_agent_index: int = field(default=0, init=False)

    def __post_init__(self):
        if len(self.agents) < 2:
            raise ValueError("At least two agents must be provided")

        instruction: str = ""

        if self.use_markdown:
            instruction += (
                "\n\nYou may use Markdown for text formatting. "
                "Examples: *italic*, **bold**, `code`, [link](https://example.com), etc."
            )

        if self.allow_termination:
            instruction += (
                "\n\nYou may terminate the conversation with the `<TERMINATE>` token "
                "if you believe it has reached a natural conclusion. "
                "Do not include the token in your message otherwise."
            )

        # Add instructions to all agents
        for agent in self.agents:
            agent.system_prompt += instruction

    def save_conversation(self, filename: Path):
        with open(filename, "w", encoding="utf-8") as f:
            # Write agent configurations
            for i, agent in enumerate(self.agents, 1):
                _ = f.write(f"=== Agent {i} ===\n\n")
                _ = f.write(f"Name: {agent.name}\n")
                _ = f.write(f"Model: {agent.model}\n")
                _ = f.write(f"Temperature: {agent.temperature}\n")
                _ = f.write(f"Context Size: {agent.ctx_size}\n")
                _ = f.write(f"System Prompt: {agent.system_prompt}\n\n")

            # Write conversation
            _ = f.write(f"=== Conversation ===\n\n")
            for i, msg in enumerate(self._conversation_log):
                if i > 0:
                    _ = f.write("\n" + "\u2500" * 80 + "\n\n")
                _ = f.write(f"{msg['agent']}: {msg['content']}\n")

    def run_conversation(self) -> Iterator[tuple[str, Iterator[str]]]:
        """Generate an iterator of conversation responses."""

        last_message = self.initial_message

        # Handle initial message if provided
        if self.initial_message is not None:
            first_agent = self.agents[0]
            first_agent.add_message("assistant", self.initial_message)
            self._conversation_log.append(
                {"agent": first_agent.name, "content": self.initial_message}
            )
            yield (first_agent.name, iter([self.initial_message]))
            self._current_agent_index = 1  # Next agent will be index 1

        while True:
            current_agent = self.agents[self._current_agent_index]
            response_stream = current_agent.chat(last_message)
            last_message_chunks: list[str] = []

            def stream_chunks() -> Iterator[str]:
                for chunk in response_stream:
                    last_message_chunks.append(chunk)
                    yield chunk

            yield (current_agent.name, stream_chunks())

            assert next(response_stream, None) is None

            last_message = "".join(last_message_chunks).strip()
            self._conversation_log.append(
                {"agent": current_agent.name, "content": last_message}
            )

            if self.allow_termination and "<TERMINATE>" in last_message:
                break

            # Move to next agent in rotation
            self._current_agent_index = (self._current_agent_index + 1) % len(self.agents)
