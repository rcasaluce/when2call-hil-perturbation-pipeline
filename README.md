
# When2Call Tool-Description Perturbation Workflow

This repository contains an **interactive, human-reviewed workflow** to produce **meaning-preserving perturbations** of tool descriptions for the **When2Call** setting, with two rewrite styles:

- **Concise perturbation** (`style_concise`)
- **Verbose perturbation** (`style_verbose`)

The dataset format and evaluation context originate from NVIDIA’s **When2Call** benchmark. ([arXiv][1])  
Upstream reference (original dataset / benchmark repository): ([NVIDIA/When2Call][2])

## What you get (outputs)

Running the workflow typically produces:

- A **working-copy JSONL** with updated tool `description` fields (only where accepted)
- An **append-only audit log** (`.jsonl`) capturing candidates, heuristics, and operator decisions
- Optional backups (if enabled)

## Quickstart (recommended)

1) Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

2. Export the Gemini token (never stored in TOML):

```bash
export TOKEN_GEMINI="..."
```

3. Create your run config:

```bash
cp scripts/config.example.toml scripts/config.toml
```

4. Start Jupyter and run the notebook:

```bash
jupyter lab
```

Open: `scripts/variants_generation.ipynb`

> If your notebook or runner supports it, you can point to the config explicitly:
>
> ```bash
> export CONFIG_PATH="scripts/config.toml"
> ```

## Repository layout

```text
README.md

docs/
  WORKFLOW.md          # full workflow spec (policy + heuristics + audit + resumability)
  CONFIG.md            # config.toml reference + env overrides + security notes

scripts/
  variants_generation.ipynb
  config_loader.py
  config.example.toml  # template to copy into scripts/config.toml

audit/                 # resumable logs (usually gitignored)
```

## Documentation

* **Workflow spec:** `docs/WORKFLOW.md`
* **Configuration reference:** `docs/CONFIG.md`

---

[1]: https://arxiv.org/pdf/2504.18851 "When2Call: When (not) to Call Tools"
[2]: https://github.com/NVIDIA/When2Call.git "NVIDIA/When2Call (upstream)"

```
```
