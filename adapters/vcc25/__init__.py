"""Allowlisted VCC25 adapter registry."""

from .base import AssetCheck, VCC25ModelAdapter
from .lingshu import LingshuAdapter
from .perturbench import PerturBenchAdapter
from .state import StateAdapter


def get_adapter(adapter_id: str) -> VCC25ModelAdapter:
    factories = {
        "vcc25.lingshu": LingshuAdapter,
        "vcc25.state": StateAdapter,
        "vcc25.perturbench": PerturBenchAdapter,
    }
    try:
        return factories[adapter_id]()
    except KeyError as exc:
        raise ValueError(f"unknown adapter: {adapter_id!r}") from exc


__all__ = ["AssetCheck", "VCC25ModelAdapter", "get_adapter"]
