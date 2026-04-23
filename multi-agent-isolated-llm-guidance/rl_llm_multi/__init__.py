"""Isolated multi-agent pure-LLM guidance package."""

from .core import MultiAgentSectorCore
from .env import JointGuidanceEnv, MultiAgentParallelEnv
from .llm import MultiAgentThreeCallController
from .utils import AssignedRoute, RouteSpec, build_route_catalog, max_agents_possible

__all__ = [
    "AssignedRoute",
    "JointGuidanceEnv",
    "MultiAgentParallelEnv",
    "MultiAgentSectorCore",
    "MultiAgentThreeCallController",
    "RouteSpec",
    "build_route_catalog",
    "max_agents_possible",
]
