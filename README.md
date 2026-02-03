# README — When2Call Tool-Description Perturbation Workflow

## 1. Purpose and scope

This workflow produces **controlled perturbations of tool descriptions** for the **When2Call** setting, with two rewrite styles:

* **Concise perturbation** (`style_concise`)
* **Verbose perturbation** (`style_verbose`)

The goal is to rewrite each tool’s `description` field **without changing meaning**, while making the text **lexically and syntactically different**. The workflow is **interactive and human-reviewed**: all model proposals are treated as candidates, and an operator explicitly decides what gets written to the dataset.

The dataset format and evaluation context originate from the **When2Call** benchmark released by NVIDIA. ([arXiv][1])
Repository reference (upstream source):

```text
https://github.com/NVIDIA/When2Call.git
```

---

## 2. What “perturbation” means here

A *perturbation* is a **meaning-preserving rewrite** of a tool description:

* **No capability drift**: no new actions, limitations, benefits, motivations, or implied steps may be introduced.
* **No deletion of information**: anything explicitly present in the base description must remain present.
* **No injection of parameters or fields**: no new parameter names, flags, IDs, or implementation details may be added.
* **Same subject and scope**: the tool being described remains the same tool, with the same scope.

The workflow enforces this via:

1. **Strict rewrite instructions** given to the LLM.
2. **Deterministic indicators and risk heuristics** computed on every candidate.
3. **Human-in-the-loop gating** (accept/edit/manual/skip).

---

## 3. Modes: concise vs verbose perturbation

### 3.1 `style_concise` (concise perturbation)

**Intent:** produce a shorter, tighter paraphrase.

Constraints enforced in the prompt:

* **1 sentence preferred** (2 maximum).
* **Shorter than the base description is preferred**.
* **If the base is already short**, the rewrite **must not exceed base length**.
* Redundancy, filler, and hedging should be removed **without losing any explicitly stated details**.

Additional policy support:

* An optional **soft target length** is computed and (when applicable) shown to the model as guidance.

### 3.2 `style_verbose` (verbose perturbation)

**Intent:** produce a paraphrase that remains compact but slightly more explicit.

Constraints enforced in the prompt:

* **1–2 sentences**, “verbose but controlled”.
* Still **no new information** and **no deletions**.
* Still **output only the rewritten description**.

### 3.3 Regeneration diversity constraint (both modes)

When regenerating candidates, an explicit diversity instruction is applied:

* The rewrite must remain meaning-equivalent.
* The **same sentence skeleton** and distinctive phrases **must not be reused**.
* The previous rewrite (or prior candidate) is provided as a “do-not-reuse” reference.

---

## 4. High-level workflow

For each tool instance in the dataset:

1. **Load base description**

   * Handles two representations:

     * Standard JSON objects (`{"description": "..."}`)
     * Tools stored as raw JSON strings (patched carefully without reserialization by default).

2. **Generate K candidates**

   * K is configurable (`NUM_CANDIDATES`, default `2`).
   * Temperature is `0.0` for stability.
   * Optional seed is passed when supported; metadata records whether it was applied.

3. **Compute deterministic indicators & risk heuristics**

   * Every candidate is scored and annotated with:

     * Length metrics
     * Lexical/structural token diffs
     * Similarity ratio
     * Risk label + reasons
     * Optional semantic signals (disabled by default)

4. **Human decision point (human-in-the-loop)**

   * Candidate acceptance is explicit.
   * Alternatives:

     * Edit a candidate
     * Provide manual rewrite
     * Regenerate another K candidates
     * Skip
     * Quit safely (including Esc)

5. **Persist decisions (append-only audit log)**

   * Every generation and decision is written to a resumable `.jsonl` audit file.
   * Resume is driven solely by the audit: previously-decided instances are skipped automatically.

6. **Apply decisions to the working dataset**

   * Writes the updated JSONL.
   * Leaves entries unchanged when skipped or when patching fails.

---

## 5. Human-in-the-loop policy (source of truth)

The workflow is explicitly designed so that **automatic acceptance never occurs**.

* Candidates are **proposals only**.
* The operator must choose among:

  * **Accept** a candidate (by index or default `ENTER` for candidate 1)
  * **Edit** a candidate into a final form
  * **Manual** entry (operator-supplied final description)
  * **Skip** (leave unchanged)
  * **Quit** (stop the session while preserving resumability)

The computed metrics are **decision support** and do not block acceptance. The intent is to keep the decision boundary human-controlled, especially for borderline paraphrases where automated heuristics can be overly conservative or permissive.

---

## 6. Candidate-generation “perturbations” visibility

During candidate generation, the workflow prints a **perturbation context block** (when enabled) that makes it clear what differs across candidates and regeneration rounds without dumping the entire prompt.

The printed context includes:

* Tool name
* Candidate index out of K
* Generation round and regeneration index
* Mode (`style_concise` or `style_verbose`)
* Seed (if configured)
* Max token limit
* Length guidance status (concise soft target applied or not, and why)
* Diversity instruction (for regeneration rounds)
* Previous rewrite hint (length + SHA snippet) used as “do-not-reuse” anchor

This improves reviewability and reduces ambiguity about why candidates differ.

---

## 7. Interactive controls and Esc-safe input

### 7.1 Commands

At each tool instance, the following commands are available:

* `ENTER` or `ok`: accept candidate **#1**
* `1..K`: accept candidate by index
* `r`: regenerate **K** new candidates
* `e`: edit an existing candidate into a final description
* `m`: manual final description
* `s`: skip (leave unchanged)
* `q`: quit
* `Esc`: quit (captured safely when raw-key mode is enabled)
* `p<idx>`: preview a candidate in full (e.g., `p2`)

### 7.2 Raw-key input (Esc behavior)

When enabled, raw-key input captures `Esc` explicitly so it behaves like quit and **cannot be misinterpreted as acceptance**. This prevents accidental acceptance when exiting the selection prompt.

---

## 8. What is measured and why

The workflow reports deterministic indicators to help detect **semantic drift risk** and **format regressions**, especially in tool descriptions that contain structured tokens (flags, identifiers, field-like patterns, numeric constraints).

### 8.1 Core length metrics

Let:

* `base_len_chars = len(base_text)`
* `cand_len_chars = len(candidate_text)`

Then:

* `len_ratio = cand_len_chars / base_len_chars` (if base length > 0)
* `len_delta_chars = cand_len_chars - base_len_chars`
* `len_delta_ratio = len_delta_chars / base_len_chars` (if base length > 0)

These are computed for every candidate and stored in the audit log.

### 8.2 Similarity metric

A character-level similarity score is computed via:

* `similarity_ratio = difflib.SequenceMatcher(None, base, cand).ratio()`

This is a **lexical similarity proxy** (not semantic equivalence). It is used only for heuristic flags (e.g., “new risky verbs + very low similarity”).

### 8.3 Indicator token extraction (deterministic regexes)

For both base and candidate, the workflow extracts tokens in these categories:

**Structural / format-sensitive**

* `flags`: tokens matching `--flag-like` patterns
* `snake`: snake_case identifiers
* `camel`: camelCase identifiers
* `field_like`: tokens that look like fields, e.g. `name:` or `name=`
* `numbers`: numeric literals (e.g., `10`, `3.14`)
* `number_units`: numbers with units (e.g., `10ms`, `5 days`)

**Semantics-sensitive surface cues**

* `verbs`: occurrences of a fixed list of “high-risk” operational verbs (create/delete/update/etc.)
* `logic`: negation and constraint words/phrases (e.g., `only`, `must`, `never`, `unless`, `at most`, `do not`)
* `modals`: modal verbs (`may`, `must`, `should`, etc.)
* `scope`: scope-like phrases linked to outputs (e.g., `returns`, `may return`, `does not return`)

Each category is extracted deterministically and stored in both raw form and preview form.

### 8.4 Token diffs

For each category `k`, set diffs are computed:

* `new[k] = cand_tokens[k] - base_tokens[k]`
* `missing[k] = base_tokens[k] - cand_tokens[k]`

These diffs are printed and written to audit as part of candidate stats.

---

## 9. Risk labeling heuristic (structural + logic primary)

Risk labeling is a heuristic triage to focus human attention:

### 9.1 Structural keys

`flags`, `field_like`, `numbers`, `number_units`, `snake`, `camel`

### 9.2 Logic keys

`logic`, `modals`, `scope`

### 9.3 Risk rules

* **HIGH** if any *new structural tokens* are introduced
  Rationale: new flags/fields/IDs/numbers often represent meaning changes or parameter injection.

* **HIGH** if any *logic/modals/scope* tokens are changed (new or missing)
  Rationale: negation/quantifiers/modals frequently invert or shift constraints.

* **HIGH** if many structural tokens are missing (threshold: `missing_structural >= 4`)
  Rationale: dropping multiple structured markers is likely an information loss.

* **MED** if some structural tokens are missing (`missing_structural > 0`) and no HIGH rule fired
  Rationale: may be acceptable but needs attention.

* **MED** if new “high-risk verbs” appear and similarity is low (`sim < 0.55`) and no HIGH rule fired
  Rationale: operational verbs can materially change the described capability.

* **MED** if similarity is very low (`sim < 0.45`) and no HIGH rule fired
  Rationale: substantial rewrites deserve extra scrutiny.

* **LOW** otherwise.

This is intentionally conservative: it is easier to flag questionable candidates than to miss subtle drift.

---

## 10. Style-specific length policy (concise soft target)

A **soft length target** exists only to guide concise perturbations in longer descriptions.

### 10.1 Parameters

* `CONCISE_TARGET_RATIO` (default `0.70`)
* `CONCISE_TARGET_MIN_BASE_LEN` (default `160`)
* `CONCISE_TARGET_MIN_CHARS` (default `80`)

### 10.2 When the soft target applies

Soft target is applied only if all are true:

1. Mode is `style_concise`
2. `base_len_chars >= CONCISE_TARGET_MIN_BASE_LEN`
3. `target_chars < base_len_chars` after applying ratio/min constraints

Target computation:

* `raw_target = int(base_len_chars * CONCISE_TARGET_RATIO)`
* `target_chars = max(raw_target, CONCISE_TARGET_MIN_CHARS)`

If `target_chars >= base_len_chars`, the target is **not applied** (reason recorded).

### 10.3 Enforcement model

The soft target is **guidance**, not a hard constraint:

* Exceeding the target is permitted **only when strictly necessary** to preserve meaning.
* Exceeding the target is flagged as a **soft flag** (`concise_exceeds_soft_target`).

### 10.4 Additional concise-mode soft flags

* Candidate longer than base: `concise_length_exceeds_base`
* Candidate has >2 sentences: `concise_sentence_count_exceeds_2`

Verbose mode similarly flags candidates with >2 sentences.

---

## 11. Optional semantic signals (disabled by default)

Two optional semantic checks can augment the deterministic indicators:

### 11.1 Embedding cosine similarity

If enabled and configured:

* Embeddings are computed for base and candidate.
* Cosine similarity is computed:

  `cosine = dot(a, b) / (||a|| * ||b||)`

A low cosine below `EMBEDDING_LOW_COSINE_THRESHOLD` raises a soft flag:

* `embedding_low_cosine`

Errors raise:

* `embedding_error`

### 11.2 Entailment verifier (LLM gate)

If enabled, a small verifier call returns exactly:

* `ENTAILS` or `NOT_ENTAILS` (or `UNKNOWN` if unexpected output)

`NOT_ENTAILS` raises a soft flag:

* `verifier_not_entails`

Verifier errors raise:

* `verifier_error`

These signals are **assistive**: they do not block acceptance automatically.

---

## 12. Audit log, resumability, and provenance

### 12.1 Append-only audit

All major actions are written as JSON lines:

* `run_start`, `run_resume`, `candidates_generated`, `regenerate`, `decision`, `patch_fallback_reserialize`, `run_end`

### 12.2 Stable instance identity

Each tool occurrence is identified by:

* `record_id`: SHA-256 of the record with tool descriptions excluded (stable under rewrites)
* `tool_index`: position inside the record’s `tools` list
* `instance_key`: `rec:<record_id>:t<tool_index>:<tool_fingerprint>`, where `tool_fingerprint` hashes the tool object excluding `description`

This makes sessions **resumable** and keeps decisions aligned to the correct tool instance even after partial edits.

### 12.3 What gets recorded

For each candidate set and decision, the audit stores:

* Base text and candidate SHA-256 + length
* Risk label and reasons
* Length metrics
* Token diffs and previews
* Optional semantic signals (if enabled)
* Model metadata (finish reason, token usage, seed applied indicator)

This supports later reporting and debugging without relying on terminal transcripts.

---

## 13. Configuration knobs (environment variables)

Key configuration points:

### Input / output selection

* `INPUT_JSONL` — input dataset JSONL
* `OUTPUT_JSONL` — working copy output path (if not set, derived from input + mode)

### Mode and model

* `MODE_KEY` — `style_concise` or `style_verbose`
* `LLM_MODEL` — model name (default: `gemini-2.5-flash`)
* `TOKEN_GEMINI` — API token (required)

### Candidate generation

* `NUM_CANDIDATES` — K candidates per tool (default `2`)
* `GEMINI_SEED` — optional seed; may be ignored by provider
* `GEMINI_MAX_TOKENS` — max tokens per generation (default `512`)
* `GEMINI_RETRY_MAX_TOKENS` — retry budget if the first output is truncated
* `MIN_SLEEP_SEC_BETWEEN_CALLS` — throttling between calls

### Length policy (concise)

* `CONCISE_TARGET_RATIO`
* `CONCISE_TARGET_MIN_BASE_LEN`
* `CONCISE_TARGET_MIN_CHARS`

### Visibility / interaction safety

* `SHOW_PERTURBATIONS` — prints perturbation context for each candidate generation
* `RAW_KEY_INPUT` — enables Esc-safe raw key handling on TTY

### Optional semantic signals

* `ENABLE_EMBEDDINGS`
* `EMBEDDING_MODEL`
* `EMBEDDING_LOW_COSINE_THRESHOLD`
* `ENABLE_VERIFIER`
* `VERIFIER_MODEL`
* `VERIFIER_MAX_TOKENS`

---

## 14. Dataset assumptions

The workflow assumes the dataset JSONL is available at:

```text
When2Call/data/test/when2call_test_llm_judge.jsonl
```


## 15. Implementation and usage

### 15.1 Implementation artifact

The workflow is implemented as a Jupyter notebook:

* `variants_generation.ipynb`

The notebook contains the full interactive rewrite pipeline (candidate generation, metrics, human decisions, audit logging, and dataset patching).

### 15.2 Prerequisites

* Python 3.10+ (recommended)
* Network access to the Gemini OpenAI-compatible endpoint:
  `https://generativelanguage.googleapis.com/v1beta/openai/`
* An API token provided via the `TOKEN_GEMINI` environment variable (required)

### 15.3 Dependencies (requirements)

A minimal `requirements.txt` sufficient for the workflow logic:

```txt
openai>=2.0.0
```

Since execution occurs in a notebook, the following are typically installed as well:

```txt
jupyterlab>=4.0.0
ipykernel>=6.0.0
```

### 15.4 Installation

Example virtual environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 15.5 Running the notebook

Environment variables must be exported before starting the notebook kernel (at minimum `TOKEN_GEMINI`).

Example:

```bash
export TOKEN_GEMINI="..."
export MODE_KEY="style_concise"   # or "style_verbose"
export LLM_MODEL="gemini-2.5-flash"
jupyter lab
```

Then open and run:

* `variants_generation.ipynb`

---
[1]: https://arxiv.org/pdf/2504.18851 "When2Call: When (not) to Call Tools"

