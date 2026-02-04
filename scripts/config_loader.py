# config_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import os

def _load_toml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any]
    try:
        import tomllib  # py3.11+
        with path.open("rb") as f:
            data = tomllib.load(f)
    except ModuleNotFoundError:
        import tomli  # pip install tomli (solo se <3.11)
        with path.open("rb") as f:
            data = tomli.load(f)
    if not isinstance(data, dict):
        raise ValueError("TOML root must be a table/object")
    return data

def _as_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Invalid float for '{name}': {x!r}")

def _as_int(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception:
        raise ValueError(f"Invalid int for '{name}': {x!r}")

def _as_bool(x: Any, name: str) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    raise ValueError(f"Invalid bool for '{name}': {x!r}")

def _as_str(x: Any, name: str) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)

def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)

@dataclass(frozen=True)
class Config:
    # paths
    input_jsonl: str
    output_jsonl: str
    audit_dir: str

    # run
    mode_key: str
    tool_field: str
    create_backup_of_target: bool

    # llm
    base_url: str
    model: str
    token_env: str
    seed: Optional[int]
    max_tokens: int
    retry_on_length: bool
    retry_max_tokens: int

    # io/behavior
    allow_reserialize_fallback: bool
    min_sleep_sec_between_calls: float

    # candidates/stats
    num_candidates: int
    candidate_snippet_chars: int
    stats_max_token_preview: int
    stats_max_token_string_len: int

    # visibility
    show_perturbations: bool
    raw_key_input: bool

    # length policy
    concise_target_ratio: float
    concise_target_min_base_len: int
    concise_target_min_chars: int

    # semantic
    enable_embeddings: bool
    embedding_model: str
    embedding_low_cosine_threshold: float
    enable_verifier: bool
    verifier_model: str
    verifier_max_tokens: int

    # styles
    styles: Dict[str, Dict[str, Any]]
    style_aliases: Dict[str, str]

def load_config(path: str) -> Config:
    p = Path(path)
    _require(p.exists(), f"Config file not found: {p}")
    raw = _load_toml(p)

    # ---- Read base values from config ----
    input_jsonl  = _as_str(_get(raw, "paths.input_jsonl"), "paths.input_jsonl")
    output_jsonl = _as_str(_get(raw, "paths.output_jsonl", ""), "paths.output_jsonl")
    audit_dir    = _as_str(_get(raw, "paths.audit_dir", "audit"), "paths.audit_dir")

    mode_key = _as_str(_get(raw, "run.mode_key", "style_concise"), "run.mode_key").strip() or "style_concise"
    tool_field = _as_str(_get(raw, "run.tool_field", "tools"), "run.tool_field") or "tools"
    create_backup = _as_bool(_get(raw, "run.create_backup_of_target", False), "run.create_backup_of_target")

    base_url  = _as_str(_get(raw, "llm.base_url"), "llm.base_url")
    model     = _as_str(_get(raw, "llm.model"), "llm.model")
    token_env = _as_str(_get(raw, "llm.token_env", "TOKEN_GEMINI"), "llm.token_env") or "TOKEN_GEMINI"

    seed_raw = _get(raw, "llm.seed", "")
    seed: Optional[int] = None
    if isinstance(seed_raw, int):
        seed = seed_raw
    elif isinstance(seed_raw, str) and seed_raw.strip():
        seed = _as_int(seed_raw.strip(), "llm.seed")

    max_tokens = _as_int(_get(raw, "llm.max_tokens", 512), "llm.max_tokens")
    retry_on_length = _as_bool(_get(raw, "llm.retry_on_length", True), "llm.retry_on_length")
    retry_max_tokens = _as_int(_get(raw, "llm.retry_max_tokens", 1024), "llm.retry_max_tokens")

    allow_reserialize = _as_bool(_get(raw, "llm.allow_reserialize_fallback", False), "llm.allow_reserialize_fallback")
    min_sleep = _as_float(_get(raw, "llm.min_sleep_sec_between_calls", 0.0), "llm.min_sleep_sec_between_calls")

    num_candidates = _as_int(_get(raw, "llm.num_candidates", 2), "llm.num_candidates")
    candidate_snippet_chars = _as_int(_get(raw, "llm.candidate_snippet_chars", 160), "llm.candidate_snippet_chars")

    stats_max_token_preview = _as_int(_get(raw, "llm.stats_max_token_preview", 8), "llm.stats_max_token_preview")
    stats_max_token_string_len = _as_int(_get(raw, "llm.stats_max_token_string_len", 48), "llm.stats_max_token_string_len")

    show_perturbations = _as_bool(_get(raw, "llm.show_perturbations", True), "llm.show_perturbations")
    raw_key_input = _as_bool(_get(raw, "llm.raw_key_input", True), "llm.raw_key_input")

    concise_target_ratio = _as_float(_get(raw, "length_policy.concise_target_ratio", 0.70), "length_policy.concise_target_ratio")
    concise_target_min_base_len = _as_int(_get(raw, "length_policy.concise_target_min_base_len", 160), "length_policy.concise_target_min_base_len")
    concise_target_min_chars = _as_int(_get(raw, "length_policy.concise_target_min_chars", 80), "length_policy.concise_target_min_chars")

    enable_embeddings = _as_bool(_get(raw, "semantic.enable_embeddings", False), "semantic.enable_embeddings")
    embedding_model = _as_str(_get(raw, "semantic.embedding_model", ""), "semantic.embedding_model").strip()
    embedding_low_cosine_threshold = _as_float(
        _get(raw, "semantic.embedding_low_cosine_threshold", 0.85),
        "semantic.embedding_low_cosine_threshold",
    )

    enable_verifier = _as_bool(_get(raw, "semantic.enable_verifier", False), "semantic.enable_verifier")
    verifier_model = _as_str(_get(raw, "semantic.verifier_model", ""), "semantic.verifier_model").strip()
    verifier_max_tokens = _as_int(_get(raw, "semantic.verifier_max_tokens", 16), "semantic.verifier_max_tokens")

    styles = _get(raw, "styles", {})
    if not isinstance(styles, dict):
        styles = {}
    style_aliases = _get(raw, "styles.aliases", {})
    if not isinstance(style_aliases, dict):
        style_aliases = {}

    # ---- Validations (hard) ----
    _require(input_jsonl.strip(), "paths.input_jsonl is required")
    _require(base_url.strip(), "llm.base_url is required")
    _require(model.strip(), "llm.model is required")

    _require(max_tokens > 0, "llm.max_tokens must be > 0")
    _require(retry_max_tokens >= max_tokens, "llm.retry_max_tokens must be >= llm.max_tokens")
    _require(num_candidates >= 1, "llm.num_candidates must be >= 1")

    _require(0.10 <= concise_target_ratio <= 1.0, "length_policy.concise_target_ratio must be in [0.10, 1.0]")
    _require(concise_target_min_base_len >= 0, "length_policy.concise_target_min_base_len must be >= 0")
    _require(concise_target_min_chars >= 0, "length_policy.concise_target_min_chars must be >= 0")

    if enable_embeddings:
        _require(embedding_model, "semantic.embedding_model must be set when enable_embeddings=true")
        _require(0.0 < embedding_low_cosine_threshold <= 1.0, "semantic.embedding_low_cosine_threshold must be in (0,1]")

    _require(verifier_max_tokens > 0, "semantic.verifier_max_tokens must be > 0")

    # ---- Env overrides (compatibilità con i tuoi ENV esistenti) ----
    # Se vuoi mantenerli identici al tuo script:
    env_mode = os.getenv("MODE_KEY")
    if env_mode and env_mode.strip():
        mode_key = env_mode.strip()

    env_model = os.getenv("LLM_MODEL")
    if env_model and env_model.strip():
        model = env_model.strip()

    env_seed = os.getenv("GEMINI_SEED")
    if env_seed and env_seed.strip():
        seed = _as_int(env_seed.strip(), "GEMINI_SEED")

    env_max_tokens = os.getenv("GEMINI_MAX_TOKENS")
    if env_max_tokens and env_max_tokens.strip():
        max_tokens = _as_int(env_max_tokens.strip(), "GEMINI_MAX_TOKENS")

    env_retry_max = os.getenv("GEMINI_RETRY_MAX_TOKENS")
    if env_retry_max and env_retry_max.strip():
        retry_max_tokens = _as_int(env_retry_max.strip(), "GEMINI_RETRY_MAX_TOKENS")

    env_allow = os.getenv("ALLOW_RESERIALIZE_FALLBACK")
    if env_allow and env_allow.strip():
        allow_reserialize = _as_bool(env_allow.strip(), "ALLOW_RESERIALIZE_FALLBACK")

    env_k = os.getenv("NUM_CANDIDATES")
    if env_k and env_k.strip():
        num_candidates = _as_int(env_k.strip(), "NUM_CANDIDATES")

    env_sleep = os.getenv("MIN_SLEEP_SEC_BETWEEN_CALLS")
    if env_sleep and env_sleep.strip():
        min_sleep = _as_float(env_sleep.strip(), "MIN_SLEEP_SEC_BETWEEN_CALLS")

    env_prev = os.getenv("STATS_MAX_TOKEN_PREVIEW")
    if env_prev and env_prev.strip():
        stats_max_token_preview = _as_int(env_prev.strip(), "STATS_MAX_TOKEN_PREVIEW")

    env_toklen = os.getenv("STATS_MAX_TOKEN_STRING_LEN")
    if env_toklen and env_toklen.strip():
        stats_max_token_string_len = _as_int(env_toklen.strip(), "STATS_MAX_TOKEN_STRING_LEN")

    env_snip = os.getenv("CANDIDATE_SNIPPET_CHARS")
    if env_snip and env_snip.strip():
        candidate_snippet_chars = _as_int(env_snip.strip(), "CANDIDATE_SNIPPET_CHARS")

    env_ratio = os.getenv("CONCISE_TARGET_RATIO")
    if env_ratio and env_ratio.strip():
        concise_target_ratio = _as_float(env_ratio.strip(), "CONCISE_TARGET_RATIO")

    env_min_base = os.getenv("CONCISE_TARGET_MIN_BASE_LEN")
    if env_min_base and env_min_base.strip():
        concise_target_min_base_len = _as_int(env_min_base.strip(), "CONCISE_TARGET_MIN_BASE_LEN")

    env_min_chars = os.getenv("CONCISE_TARGET_MIN_CHARS")
    if env_min_chars and env_min_chars.strip():
        concise_target_min_chars = _as_int(env_min_chars.strip(), "CONCISE_TARGET_MIN_CHARS")

    env_emb = os.getenv("ENABLE_EMBEDDINGS")
    if env_emb and env_emb.strip():
        enable_embeddings = _as_bool(env_emb.strip(), "ENABLE_EMBEDDINGS")

    env_emb_model = os.getenv("EMBEDDING_MODEL")
    if env_emb_model is not None:
        embedding_model = env_emb_model.strip()

    env_emb_thr = os.getenv("EMBEDDING_LOW_COSINE_THRESHOLD")
    if env_emb_thr and env_emb_thr.strip():
        embedding_low_cosine_threshold = _as_float(env_emb_thr.strip(), "EMBEDDING_LOW_COSINE_THRESHOLD")

    env_ver = os.getenv("ENABLE_VERIFIER")
    if env_ver and env_ver.strip():
        enable_verifier = _as_bool(env_ver.strip(), "ENABLE_VERIFIER")

    env_ver_model = os.getenv("VERIFIER_MODEL")
    if env_ver_model is not None:
        verifier_model = env_ver_model.strip()

    env_ver_tok = os.getenv("VERIFIER_MAX_TOKENS")
    if env_ver_tok and env_ver_tok.strip():
        verifier_max_tokens = _as_int(env_ver_tok.strip(), "VERIFIER_MAX_TOKENS")

    env_show = os.getenv("SHOW_PERTURBATIONS")
    if env_show and env_show.strip():
        show_perturbations = _as_bool(env_show.strip(), "SHOW_PERTURBATIONS")

    env_rawkey = os.getenv("RAW_KEY_INPUT")
    if env_rawkey and env_rawkey.strip():
        raw_key_input = _as_bool(env_rawkey.strip(), "RAW_KEY_INPUT")

    # disabilita embeddings “se manca il modello”
    if enable_embeddings and not embedding_model:
        enable_embeddings = False

    return Config(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        audit_dir=audit_dir,
        mode_key=mode_key,
        tool_field=tool_field,
        create_backup_of_target=create_backup,
        base_url=base_url,
        model=model,
        token_env=token_env,
        seed=seed,
        max_tokens=max_tokens,
        retry_on_length=retry_on_length,
        retry_max_tokens=retry_max_tokens,
        allow_reserialize_fallback=allow_reserialize,
        min_sleep_sec_between_calls=min_sleep,
        num_candidates=num_candidates,
        candidate_snippet_chars=candidate_snippet_chars,
        stats_max_token_preview=stats_max_token_preview,
        stats_max_token_string_len=stats_max_token_string_len,
        show_perturbations=show_perturbations,
        raw_key_input=raw_key_input,
        concise_target_ratio=concise_target_ratio,
        concise_target_min_base_len=concise_target_min_base_len,
        concise_target_min_chars=concise_target_min_chars,
        enable_embeddings=enable_embeddings,
        embedding_model=embedding_model,
        embedding_low_cosine_threshold=embedding_low_cosine_threshold,
        enable_verifier=enable_verifier,
        verifier_model=verifier_model,
        verifier_max_tokens=verifier_max_tokens,
        styles=styles,
        style_aliases=style_aliases,
    )
