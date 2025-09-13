from abc import ABC, abstractmethod
from typing import List

from class_registry.base import AutoRegister

from transformation_agent.TransformationAgent import TransformationAgent
from utils.Registries import AGENT_FACTORY_REGISTRY


class AbstractTransformationAgentFactory(AutoRegister(AGENT_FACTORY_REGISTRY), ABC):
    """
    Factory class for creating a list of transformation agents.
    """

    factory_name = None

    def __init__(self):
        pass

    @abstractmethod
    def _create_agents_impl(self) -> List[TransformationAgent]:
        """
        Abstract method to be implemented by subclasses to create and return a list of transformation agents.
        """
        pass

    def create_agents(self) -> List[TransformationAgent]:
        """
        Create and return a list of transformation agents.
        """
        agents = self._create_agents_impl()
        return agents
