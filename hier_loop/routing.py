"""Layer-to-layer routing: default adjacent, plus an explicit direct-connect whitelist.

Adjacent routing is the default and needs no whitelist entry.  A non-adjacent
edge may only be used when it appears in ``direct_whitelist``; otherwise the
route is rejected (``allowed=False``) so the loop can truncate instead of
silently skipping layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from . import layers


@dataclass
class RouteDecision:
    src: str
    dst: str
    allowed: bool
    route: str  # "adjacent" | "direct" | "blocked"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "allowed": self.allowed,
            "route": self.route,
            "reason": self.reason,
        }


@dataclass
class RoutingTable:
    """Holds the whitelist of direct edges and answers route queries."""

    direct_edges: Set[Tuple[str, str]] = field(default_factory=set)

    def can_direct(self, src: str, dst: str) -> bool:
        return (src, dst) in self.direct_edges

    def resolve(self, src: str, dst: str) -> RouteDecision:
        layers.validate_layer_id(src)
        layers.validate_layer_id(dst)
        if src == dst:
            return RouteDecision(src, dst, False, "blocked", "self-loop not allowed")
        if layers.is_adjacent(src, dst):
            return RouteDecision(src, dst, True, "adjacent", "adjacent default route")
        if (src, dst) in self.direct_edges:
            return RouteDecision(src, dst, True, "direct", "whitelisted direct edge")
        return RouteDecision(
            src,
            dst,
            False,
            "blocked",
            f"non-adjacent edge {src}->{dst} is not in the direct whitelist",
        )


def build_routing_table(direct_whitelist: Any = None) -> RoutingTable:
    """Build a routing table from ``direct_whitelist`` (list of ``{from, to}`` dicts)."""
    table = RoutingTable()
    entries = direct_whitelist or []
    if isinstance(entries, dict):
        # Accept a set-view like {"L5": ["L1"], ...}
        for src, dsts in entries.items():
            for dst in dsts or []:
                edge = layers.DirectEdge.from_dict({"from": src, "to": dst})
                table.direct_edges.add((edge.src, edge.dst))
        return table
    if not isinstance(entries, list):
        raise ValueError(f"direct_whitelist must be a list or dict, got {entries!r}")
    for raw in entries:
        edge = layers.DirectEdge.from_dict(raw)
        table.direct_edges.add((edge.src, edge.dst))
    return table


def default_routing_table() -> RoutingTable:
    return RoutingTable()
