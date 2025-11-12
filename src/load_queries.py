from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from agents.base import BaseAgent, PipelineState
from utils import (
    RunConfig,
    preprocess_query,
    validate_records_with_schema,
    load_json_schema,
    read_table_any,
    to_tsv,
)

class LoadQueriesAgent(BaseAgent):
    """
    Node 1: Load and lightly clean queries from a TSV (preferred) or CSV.

    Ground truth policy (when present):
      - Use `label` as the final gold annotation → rename to `human_label_gold`
      - Keep `level_1` and `level_2` as diagnostics → rename to
          `human_label_coarse` and `human_label_fine`
      - Expose a unified `human_label` field mirroring `human_label_gold`
        for downstream nodes (annotation/evaluation)
    """

    # Minimal requirement for unlabeled sets:
    REQUIRED_COLS = ["qid", "query"]

    # We keep these if present; harmless to miss some.
    DEFAULT_KEEP = [
        "qid", "query",
        "label",           # final gold label (optional)
        "level_1", "level_2",
        "data_split", "did", "url",
    ]

    def __init__(
        self,
        config: RunConfig,
        output_tsv: Optional[str] = None,        # defaults to config.output_checkpoint
        schema_path: Optional[str] = None,       # optional input row schema
    ) -> None:
        self.config = config
        self.output_tsv = output_tsv or config.output_checkpoint
        self.schema = load_json_schema(schema_path) if schema_path else None

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Found: {list(df.columns)}")

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = self.config.keep_cols or self.DEFAULT_KEEP
        cols = [c for c in keep if c in df.columns]
        return df[cols].copy()

    def _filter_split(self, df: pd.DataFrame) -> pd.DataFrame:
        if "data_split" not in df.columns:
            return df
        # allow 'all' to bypass filtering
        if self.config.split.lower() == "all":
            return df
        return df[df["data_split"].str.lower() == self.config.split.lower()].copy()

    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        k = int(self.config.sample_n or 0)
        if k <= 0:
            return df
        if k >= len(df):
            return df.sample(len(df), random_state=self.config.seed).copy()
        # Balanced by 'label' only if it exists; otherwise random
        if "label" in df.columns:
            return self._balanced_sample(df, k, by="label")
        return df.sample(k, random_state=self.config.seed).copy()

    def _balanced_sample(self, df: pd.DataFrame, k: int, by: str = "label") -> pd.DataFrame:
        """
        Stratified/balanced sample across classes in `by` (default: 'label').
        If a class has fewer rows than its quota, take all and top up from the rest.
        """
        if by not in df.columns:
            return df.sample(k, random_state=self.config.seed).copy()

        classes = (
            df[by].dropna().astype(str).str.strip().str.lower().unique().tolist()
        )
        classes = sorted(classes)
        if not classes:
            return df.sample(k, random_state=self.config.seed).copy()

        n_classes = len(classes)
        base = k // n_classes
        rem = k % n_classes
        rng = self.config.seed

        parts = []
        for i, c in enumerate(classes):
            quota = base + (1 if i < rem else 0)
            if quota <= 0:
                continue
            mask = df[by].astype(str).str.strip().str.lower() == c
            sub = df[mask]
            if sub.empty:
                continue
            parts.append(sub if len(sub) <= quota else sub.sample(quota, random_state=rng))

        out = pd.concat(parts, axis=0) if parts else df.head(0)

        # top-up if under quota due to scarcity
        if len(out) < k:
            remaining = df.drop(out.index, errors="ignore")
            if not remaining.empty:
                extra = remaining.sample(min(k - len(out), len(remaining)), random_state=rng)
                out = pd.concat([out, extra], axis=0)

        return out.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["query"] = df["query"].apply(
            lambda x: preprocess_query(
                x, lowercase=self.config.lowercase, strip_punct=self.config.strip_punct
            )
        )
        return df

    def _normalize_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # Rename columns to normalized field names (when present)
        rename_map = {}
        if "level_1" in df.columns:
            rename_map["level_1"] = "human_label_coarse"
        if "level_2" in df.columns:
            rename_map["level_2"] = "human_label_fine"
        if "label" in df.columns:
            rename_map["label"] = "human_label_gold"

        df = df.rename(columns=rename_map)

        # Expose a unified 'human_label' = gold label (only if we have gold)
        if "human_label_gold" in df.columns:
            df["human_label"] = df["human_label_gold"]

        return df.to_dict(orient="records")

    def run(self, state: PipelineState, **_) -> PipelineState:
        # 1) Read table (TSV preferred)
        df = read_table_any(self.config.dataset_path)
        if not {"qid", "query"}.issubset(set(df.columns)):
            try:
                df = pd.read_csv(self.config.dataset_path, sep="\t", header=None,
                                 names=["qid", "query"], dtype={0: str, 1: str})
            except Exception:
                pass
        self._check_columns(df)
        df = self._select_columns(df)

        # 3) NA handling on key fields
        if self.config.drop_na:
            needed = [c for c in ["qid", "query"] if c in df.columns]
            if needed:
                df = df.dropna(subset=needed)

        df = self._filter_split(df)
        df = self._maybe_sample(df)
        df = self._preprocess(df)

        records = self._normalize_records(df)

        if self.schema:
            errors = validate_records_with_schema(records, self.schema, fail_fast=False)
            if errors:
                state.setdefault("errors", []).extend(errors)

        if self.output_tsv:
            to_tsv(pd.DataFrame.from_records(records), self.output_tsv)

        meta = state.get("meta", {})
        meta.update({
            "n_records": len(records),
            "split": self.config.split,
            "dataset_path": self.config.dataset_path,
            "checkpoint_path": self.output_tsv,
            "prompt_version": self.config.prompt_version,
            "taxonomy_version": self.config.taxonomy_version,
        })
        state["meta"] = meta

        state["config"] = {
            "dataset_path": self.config.dataset_path,
            "split": self.config.split,
            "lowercase": self.config.lowercase,
            "strip_punct": self.config.strip_punct,
            "keep_cols": self.config.keep_cols,
            "drop_na": self.config.drop_na,
            "sample_n": self.config.sample_n,
            "seed": self.config.seed,
        }

        state["records"] = records
        return state
