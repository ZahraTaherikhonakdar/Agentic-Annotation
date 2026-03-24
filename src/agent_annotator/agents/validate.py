# agents/validate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re
import pandas as pd
from agents.base import BaseAgent, PipelineState
from utils import to_tsv

@dataclass
class ValidationConfig:
    output_tsv: str = "results/checkpoints/node3_validated.tsv"
    allowed_types: Tuple[str, ...] = ("navigational","factual","transactional","instrumental","abstain")
    conf_low: float = 0.60           # BELOW -> retry annotate
    min_len: int = 1
    url_regex: str = r"(https?://|www\.)"
    max_retries: int = 1             # avoid infinite loops

_URL_RE = re.compile(r"(https?://|www\.)", re.I)

class ValidateAgent(BaseAgent):
    def __init__(self, cfg: ValidationConfig) -> None:
        self.cfg = cfg

    def _flags_for_row(self, rec: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        q = str(rec.get("query","") or "").strip()
        t = (rec.get("ai_label") or "").strip().lower()
        try:
            c = float(rec.get("ai_confidence") or 0.0)
        except Exception:
            c = 0.0

        if t not in self.cfg.allowed_types:
            flags.append("invalid_label")
        if c < self.cfg.conf_low:
            flags.append("low_confidence")
        # if t == "abstain":
        #     flags.append("abstain")
        if len(q) < self.cfg.min_len:
            flags.append("short_query")
        if _URL_RE.search(q.replace(" ", "")):
            flags.append("url_like")
        return flags

    def run(self, state: PipelineState, **_) -> PipelineState:
        rows = state.get("records", [])
        if not rows:
            raise ValueError("Node 3: no records to validate.")

        out = []
        for rec in rows:
            flags = self._flags_for_row(rec)
            retry_count = int(rec.get("retry_count", 0))

            needs_retry = (
                ("low_confidence" in flags or "invalid_label" in flags)
                and retry_count < self.cfg.max_retries
            )

            row = dict(rec)
            row["flags"] = ",".join(flags) if flags else ""
            row["needs_retry"] = bool(needs_retry)
            row["retry_count"] = retry_count  # increment happens in annotate after a retry
            row["final_type"] = row.get("ai_label")  # judge later (eval-only) if you use it
            out.append(row)

        to_tsv(pd.DataFrame(out), self.cfg.output_tsv)
        state["records"] = out
        return state
