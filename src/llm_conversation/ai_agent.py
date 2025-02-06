from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import ollama

MessageRole = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: MessageRole
    content: str

@dataclass
class AIAgent:
    name: str
    model: str
    temperature: float = 0.8
    ctx_size: int = 2048
    _messages: list[Message] = field(default_factory=list)

    def __post_init__(self):
        if not self._messages:
            self._messages = []

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float,
        ctx_size: int,
        system_prompt: str,
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.ctx_size = ctx_size
        self._messages = [Message(role="system", content=system_prompt)]

    @property
    def messages(self) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in deepcopy(self._messages)]

    @property
    def system_prompt(self) -> str:
        return self._messages[0].content

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._messages[0].content = value

    def add_message(self, role: MessageRole, content: str):
        self._messages.append(Message(role=role, content=content))

    def _get_ollama_options(self) -> dict:
        return {
            "num_ctx": self.ctx_size,
            "temperature": self.temperature,
        }

    def chat(self, user_input: str | None) -> Iterator[str]:
        if user_input is not None:
            self.add_message("user", user_input)

        response_stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            options=self._get_ollama_options(),
            stream=True,
        )

        response_content = []
        for chunk in response_stream:
            content: str = chunk["message"]["content"]
            response_content.append(content)
            yield content

        self.add_message("assistant", "".join(response_content).strip())
