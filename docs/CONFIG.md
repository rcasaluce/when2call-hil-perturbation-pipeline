
# CONFIG — When2Call Perturbation Workflow

This document describes the configuration surface for the **tool-description perturbation workflow**: the recommended `config.toml` file, the environment-variable override layer, and token/security rules.

---

## 1. Configuration model (TOML as source of truth)

The workflow supports a **TOML configuration file** (recommended) loaded via `config_loader.py`, with optional environment-variable overrides for compatibility and quick experimentation.

### 1.1 Precedence

Effective configuration is resolved in this order:

1. **`config.toml`** values
2. **Environment overrides** (only for a defined set of variables, see §4)
3. **Hard defaults** (only when neither TOML nor env provides a value)

### 1.2 Token/security rule (important)

**The API token is never stored in `config.toml`.**  
The token is always read from an environment variable (default: `TOKEN_GEMINI`, configurable via `llm.token_env`).

---

## 2. Reference `config.toml` (complete example)

Save this as `config.toml` (typically at repo root). If your runner supports it, point to it via `--config` or `CONFIG_PATH`.

```toml
# =========================
# When2Call perturbation config
# =========================

[paths]
# Required: input dataset JSONL
input_jsonl = "When2Call/data/test/when2call_test_llm_judge.jsonl"

# Optional: working copy output path. If empty, a derived name is used:
# "<input_stem>.WORKING_COPY.<mode_key>.jsonl"
output_jsonl = ""

# Optional: audit directory (resumable JSONL logs)
audit_dir = "audit"


[run]
# Which rewrite style to apply
# Supported: "style_concise", "style_verbose"
mode_key = "style_concise"

# JSON field containing tool entries in each record
tool_field = "tools"

# Whether to create a one-time backup of the target JSONL (".bak")
create_backup_of_target = false


[llm]
# OpenAI-compatible Gemini endpoint base URL
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Model name (OpenAI-compatible)
model = "gemini-2.5-flash"

# Name of the environment variable that stores the API token
# (token value must be exported before running)
token_env = "TOKEN_GEMINI"

# Optional deterministic seed (may be ignored by provider)
seed = 12345

# Generation budget
max_tokens = 512

# Retry with a larger budget only if output was truncated (finish_reason == "length")
retry_on_length = true
retry_max_tokens = 1024

# If raw JSON-string patching fails, allow fallback by decoding+reserializing tool entry
# (default false to preserve upstream formatting)
allow_reserialize_fallback = false

# Throttle between calls (seconds)
min_sleep_sec_between_calls = 0.0

# Candidates per tool instance
num_candidates = 2

# Candidate snippet length printed in overview
candidate_snippet_chars = 160

# Stats printing limits
stats_max_token_preview = 8
stats_max_token_string_len = 48

# Visibility / interaction safety
show_perturbations = true
raw_key_input = true


[length_policy]
# Soft target for concise style (guidance only)
concise_target_ratio = 0.70
concise_target_min_base_len = 160
concise_target_min_chars = 80


[semantic]
# Optional semantic signals (disabled by default)

enable_embeddings = false
embedding_model = ""                  # must be set if enable_embeddings = true
embedding_low_cosine_threshold = 0.85

enable_verifier = false
verifier_model = ""                   # empty means "use llm.model"
verifier_max_tokens = 16


[styles]
# NOTE:
# The notebook ships with in-code STYLE_SPECS for style_verbose/style_concise.
# The [styles] table is kept for forward compatibility / extensions.
# The alias map below is actively used.

[styles.aliases]
# Optional: tolerate alternate names / typos / external naming
concise = "style_concise"
verbose  = "style_verbose"
style_coicnoso  = "style_concise"
style_coinceise = "style_concise"
````

---

## 3. Required vs optional fields

### 3.1 Required in TOML

* `paths.input_jsonl`
* `llm.base_url`
* `llm.model`

### 3.2 Required in environment at runtime

* the token env referenced by `llm.token_env` (default `TOKEN_GEMINI`)

Everything else can fall back to defaults.

---

## 4. Environment overrides (compatibility layer)

The loader supports overriding selected TOML fields via environment variables:

* `MODE_KEY` → `run.mode_key`
* `LLM_MODEL` → `llm.model`
* `GEMINI_SEED` → `llm.seed`
* `GEMINI_MAX_TOKENS` → `llm.max_tokens`
* `GEMINI_RETRY_MAX_TOKENS` → `llm.retry_max_tokens`
* `ALLOW_RESERIALIZE_FALLBACK` → `llm.allow_reserialize_fallback`
* `NUM_CANDIDATES` → `llm.num_candidates`
* `MIN_SLEEP_SEC_BETWEEN_CALLS` → `llm.min_sleep_sec_between_calls`
* `STATS_MAX_TOKEN_PREVIEW` → `llm.stats_max_token_preview`
* `STATS_MAX_TOKEN_STRING_LEN` → `llm.stats_max_token_string_len`
* `CANDIDATE_SNIPPET_CHARS` → `llm.candidate_snippet_chars`
* `CONCISE_TARGET_RATIO` → `length_policy.concise_target_ratio`
* `CONCISE_TARGET_MIN_BASE_LEN` → `length_policy.concise_target_min_base_len`
* `CONCISE_TARGET_MIN_CHARS` → `length_policy.concise_target_min_chars`
* `ENABLE_EMBEDDINGS` → `semantic.enable_embeddings`
* `EMBEDDING_MODEL` → `semantic.embedding_model`
* `EMBEDDING_LOW_COSINE_THRESHOLD` → `semantic.embedding_low_cosine_threshold`
* `ENABLE_VERIFIER` → `semantic.enable_verifier`
* `VERIFIER_MODEL` → `semantic.verifier_model`
* `VERIFIER_MAX_TOKENS` → `semantic.verifier_max_tokens`
* `SHOW_PERTURBATIONS` → `llm.show_perturbations`
* `RAW_KEY_INPUT` → `llm.raw_key_input`

---

## 5. Minimal run example (shell)

1. Export the token:

```bash
export TOKEN_GEMINI="..."
```

2. Optionally point to your config (if supported by your runner):

```bash
export CONFIG_PATH="config.toml"
```

3. Run via your notebook/script entrypoint (project-specific).

---

## 6. Notes on safety and reproducibility

* **Token handling:** keep tokens out of repo files; export them only in your shell/CI secret store.
* **Seeds:** `llm.seed` may be ignored by the provider; treat the run as “best-effort deterministic”.
* **Reserialization fallback:** keep `allow_reserialize_fallback = false` unless you accept formatting drift in tool JSON strings.


