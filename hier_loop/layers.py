"""Layer registry and loop-range topology for the hierarchical researcher.

The five layers are a fixed, ordered scale (macro -> micro).  They are only used
for organising and routing messages; they are never written into trajectory
records as an encoded field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

LayerId = str

# Ordered layer scale: index 0 = most macro (direction), 4 = most micro (technique).
LAYERS: Tuple[Tuple[LayerId, str], ...] = (
    ("L1", "direction"),
    ("L2", "problem"),
    ("L3", "hypothesis"),
    ("L4", "mechanism"),
    ("L5", "technique"),
)

_INDEX: Dict[LayerId, int] = {lid: i for i, (lid, _) in enumerate(LAYERS)}
_NAMES: Dict[LayerId, str] = {lid: name for lid, name in LAYERS}


def layer_ids() -> List[LayerId]:
    return [lid for lid, _ in LAYERS]


def layer_names() -> Dict[LayerId, str]:
    return dict(_NAMES)


def layer_index(layer: LayerId) -> int:
    if layer not in _INDEX:
        raise ValueError(f"unknown layer {layer!r}; expected one of {layer_ids()}")
    return _INDEX[layer]


def validate_layer_id(layer: Any) -> LayerId:
    if not isinstance(layer, str) or layer not in _INDEX:
        raise ValueError(f"invalid layer {layer!r}; expected one of {layer_ids()}")
    return layer


def is_adjacent(a: LayerId, b: LayerId) -> bool:
    return abs(layer_index(a) - layer_index(b)) == 1


def is_upward(a: LayerId, b: LayerId) -> bool:
    """True when b is more macro (smaller index) than a."""
    return layer_index(b) < layer_index(a)


@dataclass(frozen=True)
class LoopRange:
    """Inclusive contiguous layer interval ``[start, end]``."""

    start: LayerId
    end: LayerId

    def __post_init__(self) -> None:
        validate_layer_id(self.start)
        validate_layer_id(self.end)
        if layer_index(self.end) < layer_index(self.start):
            raise ValueError(
                f"loop_range must be monotonic: got start={self.start!r} end={self.end!r}"
            )

    def ids(self) -> List[LayerId]:
        i0, i1 = layer_index(self.start), layer_index(self.end)
        return [lid for lid, _ in LAYERS[i0 : i1 + 1]]

    def contains(self, layer: LayerId) -> bool:
        return layer_index(self.start) <= layer_index(layer) <= layer_index(self.end)

    def next_in_range(self, current: LayerId) -> Optional[LayerId]:
        """Next more-macro layer still inside the range (or None at the boundary)."""
        i = layer_index(current)
        if i <= layer_index(self.start):
            return None
        return LAYERS[i - 1][0]

    def prev_in_range(self, current: LayerId) -> Optional[LayerId]:
        """Next more-micro layer still inside the range (or None at the boundary)."""
        i = layer_index(current)
        if i >= layer_index(self.end):
            return None
        return LAYERS[i + 1][0]


@dataclass(frozen=True)
class DirectEdge:
    """A whitelisted non-adjacent direct edge ``from -> to``."""

    src: LayerId
    dst: LayerId

    def __post_init__(self) -> None:
        validate_layer_id(self.src)
        validate_layer_id(self.dst)
        if self.src == self.dst:
            raise ValueError(f"direct edge must not be a self loop: {self.src}->{self.dst}")
        if is_adjacent(self.src, self.dst):
            raise ValueError(
                f"direct_whitelist edge {self.src}->{self.dst} is adjacent; "
                "adjacent routing is the default and needs no whitelist entry"
            )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DirectEdge":
        try:
            src = validate_layer_id(raw.get("from"))
            dst = validate_layer_id(raw.get("to"))
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"malformed direct_whitelist entry: {raw!r}") from exc
        return cls(src=src, dst=dst)


def resolve_loop_range(raw: Any) -> LoopRange:
    """Accept ``{start, end}`` dict, ``[start, end]`` list, or full-range default."""
    if raw is None:
        return LoopRange(start="L1", end="L5")
    if isinstance(raw, dict):
        start = raw.get("start") or "L1"
        end = raw.get("end") or "L5"
        return LoopRange(start=validate_layer_id(start), end=validate_layer_id(end))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return LoopRange(start=validate_layer_id(raw[0]), end=validate_layer_id(raw[1]))
    raise ValueError(f"invalid loop_range {raw!r}; expected dict {{start, end}} or [start, end] or null")


def loop_range_ids(raw: Any) -> List[LayerId]:
    return resolve_loop_range(raw).ids()
