"""Hierarchical Looped RSI Researcher (`rsi_hier_loop`).

Layered research loops (L1 direction -> L5 technique) with:
- customisable loop range (``loop_range.start/end``),
- explicit cross-layer direct-connect whitelist (``direct_whitelist``),
- unified message contract ``rsi.msg.v1``,
- per-layer exit rules and experience-card rollup.

Pure stdlib, Python 3.10+; importable without the cluster engine.
"""

from __future__ import annotations

SCHEMA_VERSION = "rsi.hierloop/v1"
MSG_SCHEMA = "rsi.msg.v1"

from .layers import (  # noqa: E402
    LAYERS,
    LayerId,
    layer_ids,
    layer_index,
    validate_layer_id,
    is_adjacent,
    DirectEdge,
    resolve_loop_range,
    loop_range_ids,
)
from .messages import (  # noqa: E402
    new_bare_payload,
    make_msg,
    serialize_msg,
    Msg,
    Payload,
)
from .routing import (  # noqa: E402
    RouteDecision,
    RoutingTable,
    build_routing_table,
    default_routing_table,
)
from .exit_rules import (  # noqa: E402
    ExitDecision,
    evaluate_exit,
)
from . import decisions  # noqa: E402
from . import memory  # noqa: E402
from .experience import (  # noqa: E402
    ExperienceCard,
    new_experience_card,
    rollup_cards,
    capacity_signal,
)
from .loop import (  # noqa: E402
    HierarchicalLoop,
    LoopSummary,
    StepResult,
    LoopWorker,
    DeterministicWorker,
)

__all__ = [
    "SCHEMA_VERSION",
    "MSG_SCHEMA",
    "LAYERS",
    "LayerId",
    "layer_ids",
    "layer_index",
    "validate_layer_id",
    "is_adjacent",
    "DirectEdge",
    "resolve_loop_range",
    "loop_range_ids",
    "new_bare_payload",
    "make_msg",
    "serialize_msg",
    "Msg",
    "Payload",
    "RouteDecision",
    "RoutingTable",
    "build_routing_table",
    "default_routing_table",
    "ExitDecision",
    "evaluate_exit",
    "ExperienceCard",
    "new_experience_card",
    "rollup_cards",
    "capacity_signal",
    "HierarchicalLoop",
    "LoopSummary",
    "StepResult",
    "LoopWorker",
    "DeterministicWorker",
]
