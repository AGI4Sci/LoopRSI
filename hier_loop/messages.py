"""Unified message contract ``rsi.msg.v1``.

A message is an envelope (``from_layer`` / ``to_layer`` / ``kind`` / ``route``)
wrapping a payload that mirrors the Step0 trajectory five-tuple
(``context`` / ``action`` / ``outcome`` / ``cost`` / ``time``).  One record,
three envelope kinds: a layer-internal step, a layer-to-layer transfer, or a
loop exit summary.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import MSG_SCHEMA, layers  # noqa: F401  (ensures MSG_SCHEMA binding)

MSG_KINDS = ("step", "transfer", "exit")
ROUTES = ("adjacent", "direct")
EXIT_REASONS = ("converge", "budget", "threshold", "user")


@dataclass
class Payload:
    context: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    time: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "cost": self.cost,
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "Payload":
        raw = raw or {}
        return cls(
            context=raw.get("context") or {},
            action=raw.get("action") or {},
            outcome=raw.get("outcome") or {},
            cost=raw.get("cost") or {},
            time=raw.get("time") or {},
        )


def new_bare_payload() -> Payload:
    return Payload()


@dataclass
class Msg:
    msg_id: str
    from_layer: str
    to_layer: str
    kind: str
    round: int
    hop: int
    route: str
    payload: Payload = field(default_factory=Payload)
    parent_msg_id: Optional[str] = None
    exit_reason: Optional[str] = None
    created_at: Optional[str] = None

    def validate(self) -> None:
        layers.validate_layer_id(self.from_layer)
        layers.validate_layer_id(self.to_layer)
        if self.kind not in MSG_KINDS:
            raise ValueError(f"invalid msg kind {self.kind!r}; expected one of {MSG_KINDS}")
        if self.route not in ROUTES:
            raise ValueError(f"invalid route {self.route!r}; expected one of {ROUTES}")
        if self.kind == "exit" and self.exit_reason not in EXIT_REASONS:
            raise ValueError(
                f"exit message requires exit_reason in {EXIT_REASONS}, got {self.exit_reason!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": MSG_SCHEMA,
            "msg_id": self.msg_id,
            "parent_msg_id": self.parent_msg_id,
            "from_layer": self.from_layer,
            "to_layer": self.to_layer,
            "kind": self.kind,
            "round": self.round,
            "hop": self.hop,
            "route": self.route,
            "exit_reason": self.exit_reason,
            "payload": self.payload.to_dict(),
            "created_at": self.created_at or _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Msg":
        return cls(
            msg_id=raw["msg_id"],
            from_layer=raw["from_layer"],
            to_layer=raw["to_layer"],
            kind=raw["kind"],
            round=raw.get("round", 0),
            hop=raw.get("hop", 0),
            route=raw.get("route", "adjacent"),
            parent_msg_id=raw.get("parent_msg_id"),
            exit_reason=raw.get("exit_reason"),
            payload=Payload.from_dict(raw.get("payload")),
            created_at=raw.get("created_at"),
        )


def make_msg(
    from_layer: str,
    to_layer: str,
    kind: str,
    round_: int = 0,
    hop: int = 0,
    route: str = "adjacent",
    payload: Optional[Payload] = None,
    parent_msg_id: Optional[str] = None,
    exit_reason: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Msg:
    import hashlib

    m = Msg(
        msg_id="",  # filled below
        from_layer=from_layer,
        to_layer=to_layer,
        kind=kind,
        round=round_,
        hop=hop,
        route=route,
        payload=payload or Payload(),
        parent_msg_id=parent_msg_id,
        exit_reason=exit_reason,
        created_at=created_at or _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    basis = f"{from_layer}|{to_layer}|{kind}|{round_}|{hop}|{m.created_at}"
    m.msg_id = "msg_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return m


def serialize_msg(m: Msg) -> Dict[str, Any]:
    return m.to_dict()
