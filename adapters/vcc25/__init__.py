"""Allowlisted VCC25 adapter registry."""

from .base import AssetCheck, VCC25ModelAdapter
from .external import ExternalPaperAdapter
from .protocol import (
    REPRODUCTION_ACTIONS,
    REPRODUCTION_STATUSES,
    ReproductionContext,
    ReproductionEvidence,
    VCC25ReproductionAdapter,
)
from .lingshu import LingshuAdapter
from .perturbench import PerturBenchAdapter
from .state import StateAdapter


def get_adapter(adapter_id: str) -> VCC25ModelAdapter:
    factories = {
        "vcc25.lingshu": LingshuAdapter,
        "vcc25.state": StateAdapter,
        "vcc25.perturbench": PerturBenchAdapter,
        "vcc25.gears": lambda: ExternalPaperAdapter("gears", "vcc25.gears"),
        "vcc25.linear": lambda: ExternalPaperAdapter("linear_baseline", "vcc25.linear"),
        "vcc25.presage": lambda: ExternalPaperAdapter("presage", "vcc25.presage"),
        "vcc25.primeflow": lambda: ExternalPaperAdapter("primeflow", "vcc25.primeflow"),
        "vcc25.scgenept": lambda: ExternalPaperAdapter("scgenept", "vcc25.scgenept"),
        "vcc25.sclambda": lambda: ExternalPaperAdapter("sclambda", "vcc25.sclambda"),
    }
    try:
        return factories[adapter_id]()
    except KeyError as exc:
        raise ValueError(f"unknown adapter: {adapter_id!r}") from exc


__all__ = [
    "AssetCheck", "VCC25ModelAdapter", "VCC25ReproductionAdapter",
    "ReproductionContext", "ReproductionEvidence",
    "REPRODUCTION_ACTIONS", "REPRODUCTION_STATUSES", "get_adapter",
]
