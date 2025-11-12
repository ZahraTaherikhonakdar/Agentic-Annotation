from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# Optional deps
try:
    import jsonschema
except ImportError:
    jsonschema = None

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class RunConfig:
    dataset_path: str                  # path to input TSV
    split: str = "validation"          # "train" | "validation" | "test"
    lowercase: bool = True
    strip_punct: bool = True
    keep_cols: Optional[List[str]] = None  # defaults applied in agents
    drop_na: bool = True
    sample_n: Optional[int] = None
    output_dir: str = "results"
    output_checkpoint: str = "results/checkpoints/node1_loaded_all.tsv"
    prompt_version: str = "v1"
    taxonomy_version: str = "v1"
    seed: int = 42

    @staticmethod
    def from_yaml(path: str) -> "RunConfig":
        if yaml is None:
            raise RuntimeError("PyYAML not installed. pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return RunConfig(
            dataset_path=data["dataset_path"],
            split=data.get("split", "validation"),
            lowercase=bool(data.get("lowercase", True)),
            strip_punct=bool(data.get("strip_punct", True)),
            keep_cols=data.get("keep_cols"),
            drop_na=bool(data.get("drop_na", True)),
            sample_n=data.get("sample_n"),
            output_dir=data.get("output_dir", "results"),
            output_checkpoint=data.get("output_checkpoint", "results/checkpoints/node1_loaded_all.tsv"),
            prompt_version=data.get("prompt_version", "v1"),
            taxonomy_version=data.get("taxonomy_version", "v1"),
            seed=int(data.get("seed", 42)),
        )


def load_json_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_records_with_schema(
    records: Iterable[Dict[str, Any]],
    schema: Dict[str, Any],
    fail_fast: bool = False,
) -> List[str]:
    """Return list of error strings; empty => all good."""
    errors: List[str] = []
    if jsonschema is None:
        return errors
    validator = jsonschema.Draft7Validator(schema)
    for idx, rec in enumerate(records):
        errs = sorted(validator.iter_errors(rec), key=lambda e: e.path)
        if errs:
            for e in errs:
                errors.append(f"row={idx} path={'/'.join(map(str, e.path))} error={e.message}")
            if fail_fast:
                break
    return errors


_PUNCT_RE = re.compile(r"[^\w\s]")

def preprocess_query(q: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    if not isinstance(q, str):
        return q
    x = q
    if lowercase:
        x = x.lower()
    if strip_punct:
        x = _PUNCT_RE.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tsv(df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, sep="\t", index=False)


def read_table_any(path: str) -> pd.DataFrame:
    """Force TSV for .tsv; fallback to CSV only if the file endswith .csv"""
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        # Prefer TSV for your pipeline; error if unknown
        raise ValueError(f"Unsupported dataset extension for {path}. Use .tsv (preferred) or .csv.")
