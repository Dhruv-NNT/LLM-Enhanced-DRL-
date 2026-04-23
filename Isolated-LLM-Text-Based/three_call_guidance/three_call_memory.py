from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _coerce_bool(value: Any) -> bool:
    return bool(value)


def _numeric_similarity(a: float, b: float, scale: float) -> float:
    scale = max(1e-6, float(scale))
    return max(0.0, 1.0 - (abs(float(a) - float(b)) / scale))


def _weighted_average(weighted_scores: List[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in weighted_scores)
    if total_weight <= 0.0:
        return 0.0
    return sum(score * weight for score, weight in weighted_scores) / total_weight


def _fmt_angle(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{int(round(float(value))):+d}deg"


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{float(value):.1f}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "None"
    return str(int(value))


@dataclass
class MemoryFeatures:
    call_name: str
    intruder_bearing_deg: Optional[float]
    intruder_distance: Optional[float]
    destination_bearing_deg: Optional[float]
    destination_distance: Optional[float]
    predicted_loss_step: Optional[int]
    ttcp_steps: Optional[float]
    boundary_warning_active: bool
    boundary_exit_step: Optional[int]
    boundary_distance: Optional[float]
    safe_streak_steps: int
    merge_back_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_name": self.call_name,
            "intruder_bearing_deg": self.intruder_bearing_deg,
            "intruder_distance": self.intruder_distance,
            "destination_bearing_deg": self.destination_bearing_deg,
            "destination_distance": self.destination_distance,
            "predicted_loss_step": self.predicted_loss_step,
            "ttcp_steps": self.ttcp_steps,
            "boundary_warning_active": self.boundary_warning_active,
            "boundary_exit_step": self.boundary_exit_step,
            "boundary_distance": self.boundary_distance,
            "safe_streak_steps": self.safe_streak_steps,
            "merge_back_mode": self.merge_back_mode,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "MemoryFeatures":
        return MemoryFeatures(
            call_name=str(payload.get("call_name", "")),
            intruder_bearing_deg=_coerce_optional_float(payload.get("intruder_bearing_deg")),
            intruder_distance=_coerce_optional_float(payload.get("intruder_distance")),
            destination_bearing_deg=_coerce_optional_float(payload.get("destination_bearing_deg")),
            destination_distance=_coerce_optional_float(payload.get("destination_distance")),
            predicted_loss_step=_coerce_optional_int(payload.get("predicted_loss_step")),
            ttcp_steps=_coerce_optional_float(payload.get("ttcp_steps")),
            boundary_warning_active=_coerce_bool(payload.get("boundary_warning_active", False)),
            boundary_exit_step=_coerce_optional_int(payload.get("boundary_exit_step")),
            boundary_distance=_coerce_optional_float(payload.get("boundary_distance")),
            safe_streak_steps=int(_coerce_optional_int(payload.get("safe_streak_steps")) or 0),
            merge_back_mode=str(payload.get("merge_back_mode", "ALIGNMENT")),
        )


@dataclass
class MemoryCase:
    schema_version: int
    case_id: str
    run_id: str
    episode_id: str
    call_tag: str
    call_name: str
    step: int
    features: MemoryFeatures
    action_heading_change_deg: int
    outcome_label: str
    success: bool
    audit_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "call_tag": self.call_tag,
            "call_name": self.call_name,
            "step": self.step,
            "features": self.features.to_dict(),
            "action": {"heading_change_deg": self.action_heading_change_deg},
            "outcome_label": self.outcome_label,
            "success": self.success,
            "audit_summary": self.audit_summary,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "MemoryCase":
        action = payload.get("action") or {}
        return MemoryCase(
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            case_id=str(payload.get("case_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            call_tag=str(payload.get("call_tag", "")),
            call_name=str(payload.get("call_name", "")),
            step=int(_coerce_optional_int(payload.get("step")) or 0),
            features=MemoryFeatures.from_dict(payload.get("features") or {}),
            action_heading_change_deg=int(_coerce_optional_int(action.get("heading_change_deg")) or 0),
            outcome_label=str(payload.get("outcome_label", "")),
            success=_coerce_bool(payload.get("success", False)),
            audit_summary=str(payload.get("audit_summary", "")),
        )


@dataclass
class MemoryMatch:
    case: MemoryCase
    similarity: float


class ThreeCallMemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.cases: List[MemoryCase] = []
        self._buffer: List[MemoryCase] = []
        self.reload()

    def reload(self) -> None:
        self.cases = []
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    case = MemoryCase.from_dict(payload)
                except Exception:
                    continue
                if case.schema_version != SCHEMA_VERSION:
                    continue
                if not case.success:
                    continue
                self.cases.append(case)

    def buffer_case(
        self,
        *,
        call_tag: str,
        call_name: str,
        step: int,
        features: MemoryFeatures,
        heading_change_deg: int,
        audit_summary: str = "",
    ) -> None:
        self._buffer.append(
            MemoryCase(
                schema_version=SCHEMA_VERSION,
                case_id="",
                run_id="",
                episode_id="",
                call_tag=str(call_tag),
                call_name=str(call_name),
                step=int(step),
                features=features,
                action_heading_change_deg=int(heading_change_deg),
                outcome_label="",
                success=False,
                audit_summary=str(audit_summary or ""),
            )
        )

    def discard_buffer(self) -> None:
        self._buffer = []

    def commit_buffer(self, *, run_id: str, episode_id: str, outcome_label: str) -> int:
        if not self._buffer:
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing_case_ids = {case.case_id for case in self.cases}
        committed_count = 0

        with self.path.open("a", encoding="utf-8") as handle:
            for case in self._buffer:
                case_id = f"{run_id}:{episode_id}:{case.call_tag}"
                if case_id in existing_case_ids:
                    continue
                committed = MemoryCase(
                    schema_version=case.schema_version,
                    case_id=case_id,
                    run_id=str(run_id),
                    episode_id=str(episode_id),
                    call_tag=case.call_tag,
                    call_name=case.call_name,
                    step=case.step,
                    features=case.features,
                    action_heading_change_deg=case.action_heading_change_deg,
                    outcome_label=str(outcome_label),
                    success=True,
                    audit_summary=case.audit_summary,
                )
                handle.write(json.dumps(committed.to_dict(), ensure_ascii=True) + "\n")
                self.cases.append(committed)
                existing_case_ids.add(case_id)
                committed_count += 1

        self._buffer = []
        return committed_count

    def retrieve(
        self,
        query: MemoryFeatures,
        *,
        top_k: int,
        min_similarity: float,
        safe_r: float,
        turn_preview_steps: int,
        conflict_lookahead_steps: int,
        boundary_lookahead_steps: int,
        safe_streak_required: int,
    ) -> List[MemoryMatch]:
        eligible_cases = [
            case
            for case in self.cases
            if case.success and case.call_name == query.call_name
        ]

        matches: List[MemoryMatch] = []
        for case in eligible_cases:
            similarity = self._similarity_score(
                query,
                case.features,
                safe_r=safe_r,
                turn_preview_steps=turn_preview_steps,
                conflict_lookahead_steps=conflict_lookahead_steps,
                boundary_lookahead_steps=boundary_lookahead_steps,
                safe_streak_required=safe_streak_required,
            )
            if similarity >= float(min_similarity):
                matches.append(MemoryMatch(case=case, similarity=similarity))

        matches.sort(key=lambda item: (-item.similarity, item.case.step, item.case.case_id))
        return matches[: max(0, int(top_k))]

    def _similarity_score(
        self,
        query: MemoryFeatures,
        candidate: MemoryFeatures,
        *,
        safe_r: float,
        turn_preview_steps: int,
        conflict_lookahead_steps: int,
        boundary_lookahead_steps: int,
        safe_streak_required: int,
    ) -> float:
        if query.call_name != candidate.call_name:
            return 0.0

        boundary_exit_sentinel = float(boundary_lookahead_steps + 1)
        boundary_distance_scale = max(2.0 * float(safe_r), 1.0)
        intruder_distance_scale = max(2.0 * float(safe_r), 1.0)
        destination_distance_scale = max(10.0, float(safe_r) * 3.0)
        angle_scale = 45.0

        if query.call_name == "EXECUTE_TURN":
            loss_sentinel = float(turn_preview_steps + 1)
            weighted_scores = [
                (_numeric_similarity(query.intruder_bearing_deg or 0.0, candidate.intruder_bearing_deg or 0.0, angle_scale), 3.0),
                (_numeric_similarity(query.intruder_distance or 0.0, candidate.intruder_distance or 0.0, intruder_distance_scale), 2.0),
                (
                    _numeric_similarity(
                        float(query.predicted_loss_step if query.predicted_loss_step is not None else loss_sentinel),
                        float(candidate.predicted_loss_step if candidate.predicted_loss_step is not None else loss_sentinel),
                        loss_sentinel,
                    ),
                    3.0,
                ),
                (_numeric_similarity(query.destination_bearing_deg or 0.0, candidate.destination_bearing_deg or 0.0, angle_scale), 2.0),
                (_numeric_similarity(query.destination_distance or 0.0, candidate.destination_distance or 0.0, destination_distance_scale), 1.0),
                ((1.0 if query.boundary_warning_active == candidate.boundary_warning_active else 0.0), 1.0),
                (
                    _numeric_similarity(
                        float(query.boundary_exit_step if query.boundary_exit_step is not None else boundary_exit_sentinel),
                        float(candidate.boundary_exit_step if candidate.boundary_exit_step is not None else boundary_exit_sentinel),
                        boundary_exit_sentinel,
                    ),
                    1.0,
                ),
            ]
            return _weighted_average(weighted_scores)

        if query.call_name == "EMERGENCY_MANEUVER":
            loss_sentinel = float(conflict_lookahead_steps + 1)
            ttcp_sentinel = float(conflict_lookahead_steps + 1)
            weighted_scores = [
                (_numeric_similarity(query.intruder_bearing_deg or 0.0, candidate.intruder_bearing_deg or 0.0, angle_scale), 2.0),
                (_numeric_similarity(query.intruder_distance or 0.0, candidate.intruder_distance or 0.0, intruder_distance_scale), 2.0),
                (
                    _numeric_similarity(
                        float(query.predicted_loss_step if query.predicted_loss_step is not None else loss_sentinel),
                        float(candidate.predicted_loss_step if candidate.predicted_loss_step is not None else loss_sentinel),
                        loss_sentinel,
                    ),
                    3.0,
                ),
                (
                    _numeric_similarity(
                        float(query.ttcp_steps if query.ttcp_steps is not None else ttcp_sentinel),
                        float(candidate.ttcp_steps if candidate.ttcp_steps is not None else ttcp_sentinel),
                        ttcp_sentinel,
                    ),
                    2.0,
                ),
                (_numeric_similarity(query.destination_bearing_deg or 0.0, candidate.destination_bearing_deg or 0.0, angle_scale), 1.5),
                (_numeric_similarity(query.destination_distance or 0.0, candidate.destination_distance or 0.0, destination_distance_scale), 1.0),
                ((1.0 if query.boundary_warning_active == candidate.boundary_warning_active else 0.0), 1.0),
                (
                    _numeric_similarity(
                        float(query.boundary_exit_step if query.boundary_exit_step is not None else boundary_exit_sentinel),
                        float(candidate.boundary_exit_step if candidate.boundary_exit_step is not None else boundary_exit_sentinel),
                        boundary_exit_sentinel,
                    ),
                    1.0,
                ),
            ]
            return _weighted_average(weighted_scores)

        if query.call_name == "MERGE_BACK":
            safe_streak_scale = max(float(safe_streak_required), 1.0)
            weighted_scores = [
                (_numeric_similarity(query.destination_bearing_deg or 0.0, candidate.destination_bearing_deg or 0.0, angle_scale), 4.0),
                ((1.0 if query.merge_back_mode == candidate.merge_back_mode else 0.0), 2.0),
                ((1.0 if query.boundary_warning_active == candidate.boundary_warning_active else 0.0), 1.5),
                (
                    _numeric_similarity(
                        float(query.boundary_exit_step if query.boundary_exit_step is not None else boundary_exit_sentinel),
                        float(candidate.boundary_exit_step if candidate.boundary_exit_step is not None else boundary_exit_sentinel),
                        boundary_exit_sentinel,
                    ),
                    1.0,
                ),
                (
                    _numeric_similarity(
                        float(query.boundary_distance if query.boundary_distance is not None else boundary_distance_scale),
                        float(candidate.boundary_distance if candidate.boundary_distance is not None else boundary_distance_scale),
                        boundary_distance_scale,
                    ),
                    1.0,
                ),
                (
                    _numeric_similarity(
                        float(query.safe_streak_steps),
                        float(candidate.safe_streak_steps),
                        safe_streak_scale,
                    ),
                    1.0,
                ),
                (
                    _numeric_similarity(
                        float(query.destination_distance or 0.0),
                        float(candidate.destination_distance or 0.0),
                        destination_distance_scale,
                    ),
                    0.5,
                ),
            ]
            return _weighted_average(weighted_scores)

        return 0.0


def format_memory_card(index: int, match: MemoryMatch) -> str:
    features = match.case.features
    if match.case.call_name == "MERGE_BACK":
        details = (
            f"dest={_fmt_angle(features.destination_bearing_deg)} "
            f"mode={features.merge_back_mode} "
            f"bnd={_fmt_int(features.boundary_exit_step)} "
            f"warn={int(features.boundary_warning_active)} "
            f"safe={int(features.safe_streak_steps)}"
        )
    else:
        details = (
            f"intr={_fmt_angle(features.intruder_bearing_deg)} "
            f"sep={_fmt_float(features.intruder_distance)} "
            f"loss={_fmt_int(features.predicted_loss_step)} "
            f"dest={_fmt_angle(features.destination_bearing_deg)} "
            f"bnd={_fmt_int(features.boundary_exit_step)}"
        )
        if match.case.call_name == "EMERGENCY_MANEUVER":
            details += f" ttcp={_fmt_float(features.ttcp_steps)}"

    return (
        f"{index}) sim={match.similarity:.2f} | "
        f"{details} | "
        f"turn={int(match.case.action_heading_change_deg):+d} | "
        f"outcome={match.case.outcome_label}"
    )

