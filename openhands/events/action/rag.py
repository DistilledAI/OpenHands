from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class RAGAction(Action):
    query: str
    action: str = ActionType.RAG
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'I am interacting with the RAG with query:\n```\n{self.query}\n```'

    def __str__(self) -> str:
        ret = '**RAGAction**\n'
        ret += f'QUERY: {self.query}\n'
        ret += f'ACTION: {self.action}\n'
        return ret
