# agents/annotate.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from openai import OpenAI
import json, os, re, random
from agents.base import BaseAgent, PipelineState
from utils import RunConfig, load_json_schema, validate_records_with_schema, to_tsv


# =======================
# Config for Node 2
# =======================
@dataclass
class AnnotateConfig:
    # prompting
    mode: str = "zeroshot"
    fewshot_path: Optional[str] = "artifacts/examples_fewshot_v1.jsonl"
    temperature: float = 0.0

    # inference / votes
    self_consistency_votes: int = 1
    max_new_tokens: int = 100

    # backend
    provider: str = "openai"        # now default
    api_base: Optional[str] = None
    model_name: str = "gpt-4.1"#"gpt-4.1-mini"

    # I/O
    output_tsv: str = "results/checkpoints/node2_annotated.tsv"
    output_schema_path: Optional[str] = "artifacts/schema/output_label_schema.json"

    # meta
    prompt_version: str = "v1"
    taxonomy_version: str = "v1"

    # ORCAS-I label set
    allowed_types: Tuple[str, ...] = (
        "navigational", "factual", "transactional", "instrumental", "abstain"
    )


# =======================
# Prompt templates
# =======================
TAXONOMY_BLOCK = """You are performing a single-label taxonomy classification for the following query based on these categories:

- Navigational: go to or open a specific site/app/page (e.g., "facebook login", "bbc sport").
- Factual: seek facts, definitions, or knowledge (e.g., "what is backprop", "symptoms of flu").
- Transactional: intent to perform an action, purchase, subscribe, or download (e.g., "buy iphone 13", "download vscode").
- Instrumental: how-to, instructions, or tool usage (e.g., "install pandas", "how to reset iphone").
- Abstain: insufficient or ambiguous to decide confidently.

Classify the query into exactly ONE of these labels.
Return ONLY strict JSON in the following format:
{"type": "...", "confidence": 0.0}
"""


SYSTEM_ANNOTATOR_ZS = """
You are a senior query intent annotator.
Your task is to classify a single search query into one of the intent categories.

Taxonomy definitions:
- navigational → The query is to reach, open, or access a specific site, page, or app (e.g., "facebook login", "bbc sport").
- factual → The query seeks factual information, definitions, or knowledge (e.g., "what is backprop", "symptoms of flu").
- transactional → The query intends to perform an action such as buying, subscribing, downloading, or registering (e.g., "buy iphone 13", "download vscode").
- instrumental → The query is about how to do something or use a tool (e.g., "install pandas", "how to reset iphone").
- abstain → The query is ambiguous or lacks enough information to decide confidently.

Return ONLY a valid JSON object with exactly these keys:
{
  "type": "<one of: navigational, factual, transactional, instrumental, abstain>",
  "confidence": <float between 0 and 1>,
}

Do not include any text outside the JSON.
"""

# Taxonomy definitions:
# - navigational → The query is to reach, open, or access a specific site, page, or app (e.g., "facebook login", "bbc sport").
# - factual → The query seeks factual information, definitions, or knowledge (e.g., "what is backprop", "symptoms of flu").
# - transactional → The query intends to perform an action such as buying, subscribing, downloading, or registering (e.g., "buy iphone 13", "download vscode").
# - instrumental → The query is about how to do something or use a tool (e.g., "install pandas", "how to reset iphone").
# - abstain → The query is ambiguous or lacks enough information to decide confidently.


SYSTEM_ANNOTATOR_FS = """
You are a senior query intent annotator. Your task is to classify a single search query into one of the intent categories.

You will see several EXAMPLES (query → label). Use them as guidance. If examples conflict or do not cover the case, fall back to these rules:

Taxonomy definitions:
- navigational (brand/domain/URL/login/homepage) →
- transactional (buy/subscribe/register/download/apply/pay/book/reserve/renew/cancel) →
- instrumental (how to/steps/use/install/fix/reset/configure/setup/recipe/tutorial/guide) →
- factual (what/when/who/why/meaning/symptoms/definition/info) →
- abstain (insufficient/ambiguous).

Return ONLY a valid JSON object with exactly these keys:
{
  "type": "<one of: navigational, factual, transactional, instrumental, abstain>",
  "confidence": <float between 0 and 1>,
}

Do not include any text outside the JSON.
 """

def build_user_prompt_zeroshot(query_text: str) -> str:
    # Short + strict; the SYSTEM prompt carries the taxonomy/rules.
    return (
        'Task: classify the query into exactly one label and return JSON only.\n'
        'Allowed labels: navigational, factual, transactional, instrumental, abstain.\n'
        f'Query: "{query_text}"'
    )

def build_user_prompt_fewshot(query_text: str, shots: List[Dict[str, Any]]) -> str:
    # Up to 8 concise examples:  Example: "query" → label
    lines: List[str] = []
    for ex in shots[:8]:
        q = str(ex.get("query", "")).strip()
        t = str(ex.get("type", "")).strip().lower()
        if q and t:
            lines.append(f'Example: "{q}" → {t}')
    examples = "\n".join(lines)
    return (
        (examples + "\n\n") if examples else ""
    ) + 'Now classify the following query and return JSON only.\n' + f'Query: "{query_text}"'




# =======================
# JSON helpers
# =======================

# JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)  # non-greedy


# def extract_json(text: str) -> str:
#     m = JSON_BLOCK_RE.search(text or "")
#     if not m:
#         return json.dumps({"type": "abstain", "confidence": 0.0})
#     return m.group(0)
# def extract_json(text: str) -> str:
#     if not text:
#         return json.dumps({"type": "abstain", "confidence": 0.0})
#     # return the first *parseable* JSON object
#     for m in JSON_BLOCK_RE.finditer(text):
#         cand = m.group(0)
#         try:
#             json.loads(cand)
#             return cand
#         except Exception:
#             continue
#     return json.dumps({"type": "abstain", "confidence": 0.0})

# Fixed extract_json function for your annotate.py

JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)

def extract_json(text: str) -> str:
    """
    Grab the *last* parseable JSON-ish object from the model output.
    Handles standard JSON and simple Python-dict style with single quotes.
    """
    if not text:
        return json.dumps({"type": "abstain", "confidence": 0.0})

    last_good = None

    for m in JSON_BLOCK_RE.finditer(text):
        cand = m.group(0).strip()

        # 1) try as-is (proper JSON)
        try:
            json.loads(cand)
            last_good = cand
            continue
        except Exception:
            pass

        # 2) try quick single-quote → double-quote fix (Python dict style)
        if "'" in cand and '"' not in cand:
            fixed = cand.replace("'", '"')
            try:
                json.loads(fixed)
                last_good = fixed
            except Exception:
                pass

    if last_good is not None:
        return last_good

    # fallback
    return json.dumps({"type": "abstain", "confidence": 0.0})



def normalize_type(t: str, allowed: Tuple[str, ...]) -> str:
    t = (t or "").strip().lower()
    aliases = {
        "info": "factual", "informational": "factual", "informative": "factual",
        "nav": "navigational", "navigation": "navigational",
        "transact": "transactional", "purchase": "transactional",
        "howto": "instrumental", "how-to": "instrumental", "procedural": "instrumental",
    }
    t = aliases.get(t, t)
    return t if t in allowed else "abstain"


def clamp_conf(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))


# =======================
# OpenAI backend builder
# =======================
def build_openai_client(provider: str, api_base: Optional[str]) -> OpenAI:
    if provider == "openai":
        token = os.getenv("OPENAI_API_KEY")
        if not token:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        base = api_base or "https://api.openai.com/v1"
        return OpenAI(base_url=base, api_key=token)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
def openai_chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    force_json: bool = True,
    is_deepseek: bool = False,
) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }

    # IMPORTANT: do NOT send response_format to DeepSeek
    if force_json and not is_deepseek:
        kwargs["response_format"] = {"type": "json_object"}

    tries = 0
    while True:
        tries += 1
        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            # Always collect both content and reasoning_content.
            parts = []
            c = getattr(msg, "content", None)
            if c:
                parts.append(c)
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                parts.append(rc)

            raw = "\n".join(parts).strip()

            if os.getenv("DEBUG_LLM") == "1":
                print("[DEBUG][DeepSeek] raw output:", raw[:400])

            # If for some reason everything is empty, still return empty string
            return raw
        except Exception as e:
            if tries >= 2:
                print(f"[annotate] LLM error: {e}")
                return json.dumps({"type": "abstain", "confidence": 0.0})

# def openai_chat_json(
#     client: OpenAI,
#     model: str,
#     system_prompt: str,
#     user_prompt: str,
#     max_new_tokens: int,
#     temperature: float,
#     force_json: bool = True,
# ) -> str:
#     kwargs = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": float(temperature),
#         "max_tokens": int(max_new_tokens),
#     }
#     if force_json:
#         kwargs["response_format"] = {"type": "json_object"}
#
#     tries = 0
#     while True:
#         tries += 1
#         try:
#             #gpt
#             # resp = client.chat.completions.create(**kwargs)
#             # return resp.choices[0].message.content or ""
#             resp = client.chat.completions.create(**kwargs)
#             # Prefer normal content; also include DeepSeek's reasoning_content if present.
#             msg = resp.choices[0].message
#             parts = []
#             if getattr(msg, "content", None):
#                     parts.append(msg.content)
#             rc = getattr(msg, "reasoning_content", None)
#             if rc:
#                 # appending keeps backward-compat, and our extractor will pick the first valid JSON
#                 parts.append(rc)
#             return "\n".join(parts) if parts else ""
#         except Exception as e:
#             if tries >= 2:
#                 #gpt
#                 #return json.dumps({"type": "abstain", "confidence": 0.0})
#                 # Tiny, safe debug (remove later if you like)
#                 print(f"[annotate] LLM error: {e}")
#                 return json.dumps({"type": "abstain", "confidence": 0.0})


# =======================
# Main Agent
# =======================
class LLMAnnotateAgent(BaseAgent):
    """Node 2: Annotate queries using OpenAI GPT-4.1-mini."""

    def __init__(self, run_cfg: RunConfig, ann_cfg: AnnotateConfig) -> None:
        self.run_cfg = run_cfg
        self.ann_cfg = ann_cfg
        self.schema = load_json_schema(ann_cfg.output_schema_path) if ann_cfg.output_schema_path else None

        if ann_cfg.provider == "openai":
            self.client = build_openai_client("openai", ann_cfg.api_base)
        else:
            raise ValueError("Only provider='openai' is supported in this config.")

        # NEW: few-shot examples (only when requested)
        # Few-shot examples (only when requested)
        self._fewshots: List[Dict[str, Any]] = []
        if (self.ann_cfg.mode or "zeroshot").lower() == "fewshot":
            self._fewshots = self._read_jsonl(self.ann_cfg.fewshot_path)
            random.Random(42).shuffle(self._fewshots)  # stable variety

    # improved_json_extraction.py
    import json
    import re

    def extract_json_improved(text: str) -> str:
        """
        Improved JSON extraction that works better with DeepSeek reasoning models.
        """
        if not text:
            print(f"[DEBUG] DeepSeek raw response: {text}")
            return json.dumps({"type": "abstain", "confidence": 0.0})

        # Print raw response for debugging (remove in production)
        print(f"[DEBUG] Raw response: {text[:500]}...")

        # Try multiple JSON extraction strategies

        # Strategy 1: Look for complete JSON objects
        json_patterns = [
            r'\{[^{}]*"type"[^{}]*"confidence"[^{}]*\}',  # JSON with required fields
            r'\{[^{}]*"label"[^{}]*"confidence"[^{}]*\}',  # Alternative field names
            r'\{[^{}]*"classification"[^{}]*"confidence"[^{}]*\}',
            r'\{.*?\}',  # Any JSON object
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Check if it has the fields we need
                    if any(key in parsed for key in ['type', 'label', 'classification', 'intent']):
                        return match
                except json.JSONDecodeError:
                    continue

        # Strategy 2: Extract values manually if JSON parsing fails
        type_patterns = [
            r'"type"\s*:\s*"([^"]+)"',
            r'"label"\s*:\s*"([^"]+)"',
            r'"classification"\s*:\s*"([^"]+)"',
            r'"intent"\s*:\s*"([^"]+)"',
            r'type\s*:\s*([a-zA-Z]+)',
            r'label\s*:\s*([a-zA-Z]+)',
        ]

        conf_patterns = [
            r'"confidence"\s*:\s*([0-9.]+)',
            r'"conf"\s*:\s*([0-9.]+)',
            r'"score"\s*:\s*([0-9.]+)',
            r'confidence\s*:\s*([0-9.]+)',
        ]

        type_val = None
        conf_val = 0.0

        for pattern in type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                type_val = match.group(1).strip().lower()
                break

        for pattern in conf_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    conf_val = float(match.group(1))
                    break
                except ValueError:
                    continue

        if type_val:
            return json.dumps({"type": type_val, "confidence": conf_val})

        # Strategy 3: Look for intent keywords in the text
        intent_keywords = {
            'navigational': ['navigate', 'go to', 'visit', 'access', 'login', 'website'],
            'factual': ['what', 'who', 'when', 'where', 'why', 'how', 'define', 'explain'],
            'transactional': ['buy', 'purchase', 'download', 'subscribe', 'order', 'book'],
            'instrumental': ['how to', 'tutorial', 'guide', 'steps', 'install', 'setup'],
        }

        text_lower = text.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return json.dumps({"type": intent, "confidence": 0.5})

        # Default fallback
        return json.dumps({"type": "abstain", "confidence": 0.0})

    # Test the function with some sample DeepSeek responses
    if __name__ == "__main__":
        # Test cases
        test_responses = [
            '{"type": "factual", "confidence": 0.8}',
            'Based on the query, I classify this as {"type": "navigational", "confidence": 0.9}',
            'The classification is: type: factual, confidence: 0.7',
            'This appears to be a factual query with high confidence.',
            '{"label": "transactional", "confidence": 0.85}',
        ]

        for i, response in enumerate(test_responses, 1):
            result = extract_json_improved(response)
            print(f"Test {i}: {result}")

    def _one_vote(self, query_text: str) -> Dict[str, Any]:
        mode = (self.ann_cfg.mode or "zeroshot").lower()
        if mode == "fewshot" and getattr(self, "_fewshots", None):
            system_prompt = SYSTEM_ANNOTATOR_FS
            user_prompt = build_user_prompt_fewshot(query_text, self._fewshots)
        else:
            system_prompt = SYSTEM_ANNOTATOR_ZS
            user_prompt = build_user_prompt_zeroshot(query_text)

        # Detect DeepSeek from api_base or model name
        base_lower = (self.ann_cfg.api_base or "").lower()
        model_lower = (self.ann_cfg.model_name or "").lower()
        is_deepseek = ("deepseek" in base_lower) or ("deepseek" in model_lower)

        raw = openai_chat_json(
            client=self.client,
            model=self.ann_cfg.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.ann_cfg.max_new_tokens,
            temperature=self.ann_cfg.temperature,
            force_json=not is_deepseek,   # JSON mode only for OpenAI, not DeepSeek
            is_deepseek=is_deepseek,
        )

        # Optional extra debug
        if os.getenv("DEBUG_LLM") == "1":
            print("[annotate][_one_vote] raw:", (raw or "")[:300].replace("\n", " "))

        # --- parse JSON-ish output ---
        try:
            parsed = json.loads(extract_json(raw))
        except Exception:
            parsed = {"type": "abstain", "confidence": 0.0}

        # Accept common aliases the model might emit
        label_keys = ("type", "label", "class", "category", "intent", "prediction", "final_type")
        t_raw = None
        for k in label_keys:
            if k in parsed and parsed[k]:
                t_raw = str(parsed[k])
                break
        t = normalize_type(t_raw or "", self.ann_cfg.allowed_types)

        if t == "abstain" and os.getenv("DEBUG_LLM") == "1":
            print("[annotate][debug] parsed abstain from:", (raw or "")[:300].replace("\n", " "))

        # Confidence: accept confidence, conf, score, prob; allow percents
        conf_keys = ("confidence", "conf", "score", "prob", "probability", "confidence_score")
        conf_raw = None
        for k in conf_keys:
            if k in parsed:
                conf_raw = parsed[k]
                break

        conf_val = 0.0
        if isinstance(conf_raw, str) and conf_raw.strip().endswith("%"):
            try:
                conf_val = float(conf_raw.strip().rstrip("%")) / 100.0
            except Exception:
                conf_val = 0.0
        else:
            try:
                conf_val = float(conf_raw)
            except Exception:
                conf_val = 0.0

        conf = clamp_conf(conf_val)
        return {"type": t, "confidence": conf}


    def _annotate_one(self, query_text: str) -> Dict[str, Any]:
        """Runs one annotation; if self_consistency_votes>1, aggregates votes."""
        v = int(getattr(self.ann_cfg, "self_consistency_votes", 1) or 1)
        if v <= 1:
            return self._one_vote(query_text)

        votes = [self._one_vote(query_text) for _ in range(v)]

        from collections import defaultdict
        counts, confs = defaultdict(int), defaultdict(list)
        for r in votes:
            lbl = r.get("type", "abstain")
            c = float(r.get("confidence", 0.0) or 0.0)
            counts[lbl] += 1
            confs[lbl].append(c)

        # pick label by (count → mean confidence → max confidence)
        def score(lbl):
            mean_c = sum(confs[lbl]) / len(confs[lbl]) if confs[lbl] else 0.0
            max_c = max(confs[lbl]) if confs[lbl] else 0.0
            return (counts[lbl], mean_c, max_c)

        best = max(counts.keys(), key=score)
        best_conf = sum(confs[best]) / len(confs[best]) if confs[best] else 0.0
        return {"type": best, "confidence": best_conf}

    def _read_jsonl(self, path: Optional[str]) -> List[Dict[str, Any]]:
        if not path:
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                        q = str(ex.get("query", "")).strip()
                        t = str(ex.get("type", "")).strip().lower()
                        if q and t:
                            rows.append({"query": q, "type": t})
                    except Exception:
                        # skip malformed line
                        pass
        except FileNotFoundError:
            pass
        return rows

    # in agents/annotate.py, inside class LLMAnnotateAgent(BaseAgent):

    def run(self, state: PipelineState, **_) -> PipelineState:
        records = state.get("records", [])
        if not records:
            raise ValueError("Node 2: no records found in state. Did Node 1 run?")

        out_rows: List[Dict[str, Any]] = []
        for i, rec in enumerate(records, 1):
            has_label = "ai_label" in rec and rec.get("ai_label") is not None
            needs_retry = bool(rec.get("needs_retry", False))

            # annotate if no label yet (first pass), or flagged for retry
            if (not has_label) or needs_retry:
                res = self._annotate_one(rec.get("query", ""))
                row = dict(rec)
                row.update({
                    "ai_label": res["type"],
                    "ai_confidence": res["confidence"],
                    "model_name": self.ann_cfg.model_name,
                    "provider": self.ann_cfg.provider,
                    "prompt_version": self.ann_cfg.prompt_version,
                    "taxonomy_version": self.ann_cfg.taxonomy_version,
                })
                # this was a retry if it had needs_retry
                if needs_retry:
                    row["retry_count"] = int(row.get("retry_count", 0)) + 1
                # clear retry flag after annotating
                row["needs_retry"] = False
            else:
                # keep existing
                row = dict(rec)

            out_rows.append(row)
            if i % 10 == 0:
                print(f"[annotate] {i}/{len(records)} done")

        # (optional) schema validate ai_* fields
        # if self.schema:
        #     errs = validate_records_with_schema(
        #         [{"type": r.get("ai_label"), "confidence": r.get("ai_confidence", 0.0)} for r in out_rows],
        #         self.schema,
        #         fail_fast=False,
        #     )
        #     if errs:
        #         state.setdefault("errors", []).extend(errs)

        to_tsv(pd.DataFrame(out_rows), self.ann_cfg.output_tsv)
        state["records"] = out_rows
        state["annotations"] = [
            {"qid": r.get("qid"), "query": r.get("query"),
             "ai_label": r.get("ai_label"), "ai_confidence": r.get("ai_confidence")}
            for r in out_rows
        ]
        return state
