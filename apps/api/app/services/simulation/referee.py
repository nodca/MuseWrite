from __future__ import annotations

from app.services.simulation.policy import DEFAULT_POLICY, SimulationPolicy
from app.services.simulation.types import RefereeDecision, TurnCandidate


def pick_speaker(
    candidates: list[TurnCandidate],
    *,
    turn_index: int = 0,
    policy: SimulationPolicy = DEFAULT_POLICY,
) -> RefereeDecision:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.agitation_score >= policy.min_candidate_score
        and candidate.confidence >= policy.min_candidate_confidence
    ]
    if not eligible:
        if turn_index == 1 and candidates:
            winner = max(
                candidates,
                key=lambda item: (item.agitation_score, item.confidence, -item.actor_id),
            )
            return RefereeDecision(
                winner_id=winner.actor_id,
                reason="scene_open",
                applied_rules=["cold_open_override"],
                pause_suggested=False,
            )
        return RefereeDecision(
            winner_id=None,
            reason="signal_too_low",
            applied_rules=["pause_on_low_signal"],
            pause_suggested=True,
        )

    winner = max(
        eligible,
        key=lambda item: (item.agitation_score, item.confidence, -item.actor_id),
    )
    return RefereeDecision(
        winner_id=winner.actor_id,
        reason=winner.motive,
        applied_rules=["max_agitation_score"],
        pause_suggested=False,
    )

