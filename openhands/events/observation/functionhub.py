from dataclasses import dataclass, field
from typing import List

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class FunctionHubObservation(Observation):
    """This data class represents the result of a Function Hub operation."""

    observation: str = ObservationType.FUNCTION_HUB
    function_name: str = field(default='')
    id_functionhub: str = field(default='')
    text_content: str = field(default='')
    image_urls: List[str] = field(default_factory=list)
    video_urls: List[str] = field(default_factory=list)
    audio_urls: List[str] = field(default_factory=list)
    blob: str = field(default='')
    error: str = field(default='')

    @property
    def message(self) -> str:
        return self.content
