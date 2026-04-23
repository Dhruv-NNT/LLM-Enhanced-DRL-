"""Multi-agent RL + LLM prototype package."""

from .core import MultiAgentSectorCore
from .env import JointGuidanceEnv, MultiAgentParallelEnv
from .llm import MultiAgentThreeCallController
from .ppo import SharedActorCentralCriticPPO
from .utils import AssignedRoute, RouteSpec, build_route_catalog, max_agents_possible

__all__ = [
    "AssignedRoute",
    "JointGuidanceEnv",
    "MultiAgentParallelEnv",
    "MultiAgentSectorCore",
    "MultiAgentThreeCallController",
    "RouteSpec",
    "SharedActorCentralCriticPPO",
    "build_route_catalog",
    "max_agents_possible",
]

