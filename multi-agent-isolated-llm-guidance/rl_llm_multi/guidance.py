"""No-op guidance interface for future LLM/RL integration."""

from __future__ import annotations

from typing import Dict, Mapping, Optional


class NoGuidanceProvider:
    """Return no action overrides while preserving a trainer call site."""

    def reset(self) -> None:
        return None

    def actions_for_step(
        self,
        core,
        *,
        step: int,
        proposed_actions: Mapping[str, int],
    ) -> Dict[str, int]:
        return {}

    def get_advice_for_rl(self, agent_id: Optional[str] = None):
        return None
