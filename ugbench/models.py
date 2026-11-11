"""Per-model request rules for Anthropic Claude models on Amazon Bedrock.

The benchmark sends the *same* request on every path. What that request may contain depends on the model:

* **Sampling parameters.** Newer models reject ``temperature`` with a 400 (Opus 4.7+, every 5.x model).
* **Thinking.** Some models think by default. For a latency benchmark we want the shortest, least variable
  reply, so thinking is turned off where the model allows it, and kept minimal (``effort: low``) where it doesn't.

Everything is derived from the Bedrock model ID (``us.anthropic.claude-sonnet-5-5``,
``anthropic.claude-haiku-4-5-20251001-v1:0``, ``global.anthropic.claude-opus-5-5``, ...), so a new model
works without code changes. Models newer than the table fall back to the rules of the newest known generation.
Every derived value can be overridden in ``workload`` (``temperature``, ``thinking``, ``effort``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# claude-<tier>-<major>[-<minor>][-<date>][-v<n>:<m>]   (tier: haiku | sonnet | opus | fable | mythos)
_ID = re.compile(r"claude-(?P<tier>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?(?=$|[-:@])")

THINKING_DISABLED = {"type": "disabled"}
THINKING_BETWEEN_TOOLS = {"type": "between_tools"}   # Claude Sonnet 5.5: its only "thinking off" mode


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    tier: str            # haiku | sonnet | opus | fable | mythos | unknown
    version: tuple       # (major, minor), e.g. (5, 5); (0, 0) if unparseable

    @property
    def slug(self) -> str:
        """Short, UC-safe name used for service names: ``sonnet_5_5``, ``haiku_4_5``, ``opus_5``."""
        if self.tier == "unknown":
            return re.sub(r"[^a-z0-9]+", "_", self.model_id.lower()).strip("_")[:40] or "model"
        major, minor = self.version
        return f"{self.tier}_{major}" + (f"_{minor}" if minor else "")


def parse(model_id: str) -> ModelInfo:
    model_id = model_id or ""
    m = _ID.search(model_id)
    if not m:
        return ModelInfo(model_id, "unknown", (0, 0))
    return ModelInfo(model_id, m["tier"], (int(m["major"]), int(m["minor"] or 0)))


def _accepts_temperature(mi: ModelInfo) -> bool:
    if mi.tier in ("fable", "mythos"):
        return False
    if mi.tier == "unknown":
        return True                        # be permissive; a 400 in `ugbench health` says what to override
    if mi.version >= (5, 0):
        return False                       # every 5.x model
    return not (mi.tier == "opus" and mi.version >= (4, 7))   # Opus 4.7 / 4.8


def _thinking_off(mi: ModelInfo) -> tuple[dict | None, str | None]:
    """(thinking field, effort) that give the shortest reply. ``None`` = don't send the field."""
    t, v = mi.tier, mi.version
    if t in ("fable", "mythos"):
        return None, "low"                 # thinking is always on; only effort controls it
    if t == "opus" and v >= (5, 5):
        return None, "low"                 # Opus 5.5: `disabled` is rejected at every effort
    if t == "sonnet" and v >= (5, 5):
        return THINKING_BETWEEN_TOOLS, None
    if v >= (5, 0):
        return THINKING_DISABLED, None     # Opus 5, Sonnet 5, Haiku 5.5: on by default, `disabled` accepted
    return None, None                      # 4.x and older: thinking is off unless requested


def request_fields(cfg: dict) -> dict:
    """Resolve ``temperature`` / ``thinking`` / ``effort`` for the configured model.

    ``workload.temperature``: dropped if the model rejects it.
    ``workload.thinking`` / ``workload.effort``: ``"auto"`` (default) = the model's rule above;
    ``None`` = don't send; any other value is sent as given.
    """
    w = cfg["workload"]
    mi = parse(cfg["aws"]["bedrock_model_id"])
    temp = w.get("temperature")
    if temp is not None and not _accepts_temperature(mi):
        temp = None
    auto_thinking, auto_effort = _thinking_off(mi)
    thinking = w.get("thinking", "auto")
    effort = w.get("effort", "auto")
    return {"temperature": temp,
            "thinking": auto_thinking if thinking == "auto" else thinking,
            "effort": auto_effort if effort == "auto" else effort}


def describe(cfg: dict) -> str:
    f = request_fields(cfg)
    parts = [f"temperature={f['temperature']}" if f["temperature"] is not None else "temperature not sent"]
    parts.append(f"thinking={f['thinking']}" if f["thinking"] else "no thinking field")
    if f["effort"]:
        parts.append(f"effort={f['effort']}")
    mi = parse(cfg["aws"]["bedrock_model_id"])
    return f"{mi.model_id} ({mi.slug}): " + ", ".join(parts)
