# Classical Chinese Poetry Translation Experiment

This repository contains the generation, blind judging, and poem-level analysis pipeline for comparing six translation prompt conditions across three local OpenAI-compatible models. The dataset is stored at `data/poems.json`; it contains 4 development poems and 16 test poems. The formal test generation contains 864 assigned translations: 16 poems × 6 conditions × 3 models × 3 repetitions.

Read the complete Chinese [experiment report](实验报告.md) for the research questions, methods, results, and interpretation.

The experiment code and related instructions are available at [the course repository](https://github.com/stq1234/LLMandAppliedPractices-Classical-Poetry-Translation).

## Rebuild the archived report results

The completed formal run is stored in `results/generation_manifest.json`, `results/generation`, and `results/judging`. To recompute its statistics and figures without making any model calls:

```powershell
python -m pip install -r requirements.txt
python -m src.revalidate
python -m src.cli summarize --judge-results-dir results/judging_accepted
python -m src.plots
```

To generate a new independent run, use a new manifest and separate generation/judging result directories consistently in the commands below. Running with the archived directories resumes the existing tasks and keeps their accepted responses; it does not create independent new samples.

## Requirements and API configuration

Use Python 3.10 or newer. Install the runtime dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The API client only accepts a plain-HTTP localhost endpoint. By default it uses `http://localhost:8080/v1`; set `OPENAI_BASE_URL` in the current process environment only if the local server uses another localhost port. Set `OPENAI_API_KEY` in the process environment before running the commands. The key is never written to manifests or result files, and error messages redact key-shaped strings. Temperature and `top_p` are omitted. Generation requests use reasoning effort `low` and 2048 maximum completion tokens; judge requests use `low` and 8192 maximum completion tokens.

For example, set the key in the PowerShell process that will run the experiment, then start the local API server separately:

```powershell
$env:OPENAI_API_KEY = Read-Host "Local API key"
```

The three configured model IDs are `gpt-6.1-sol`, `gpt-6-luna`, and `gpt-5.6-luna`. Model aliases are not substituted automatically; a response must identify the requested model or that model ID with a valid date suffix.

## Development run

Use a separate manifest and result directory for the 72-call development pass. This prevents development attempts from colliding with a later formal run.

```powershell
python -m src.cli prepare --split dev --limit-poems 4 --repetitions 1 --manifest results/dev_v2_manifest.json
python -m src.cli run --manifest results/dev_v2_manifest.json --results-dir results/generation_dev_v2 --split dev --workers 3
```

After generation completes, the development judging pass contains 12 batches:

```powershell
python -m src.cli judge --split dev --variant primary --generation-dir results/generation_dev_v2 --generation-manifest results/dev_v2_manifest.json --results-dir results/judging_dev_v2
```

The development split is for checking prompts and the evaluation format. Its results are not part of the 16-poem formal test analysis unless explicitly selected for a separate analysis.

## Formal test run

Prepare the complete test manifest. Without `--limit-poems`, preparation validates the full 20-poem corpus split (4 dev, 16 test) and writes only the selected split's tasks.

```powershell
python -m src.cli prepare --split test --manifest results/generation_manifest.json
python -m src.cli run --manifest results/generation_manifest.json --results-dir results/generation --split test --workers 3
```

The default manifest assigns repetitions `0`, `1`, and `2` in a deterministic shuffled order. The generation default worker count is 3. The archived formal generation used 4 workers; concurrency controls scheduling, not the prompt or decoding parameters. Each call is saved as an append-only JSONL attempt record. Rerunning the same manifest and result directory resumes unfinished tasks; a previously accepted response is kept. API errors, truncation, invalid JSON, and invalid output schema can be retried up to two times. Score-based regeneration is never performed. If a stored task ID has different prompts or parameters from the current manifest, resuming stops with an error instead of mixing runs. Failed tasks remain recorded, and the command exits with a nonzero status when any selected task remains failed.

Generation outputs are stored separately by split (`dev.jsonl` and `test.jsonl`) inside the selected result directory. Every record contains the task identifiers, frozen request messages and parameters, raw normalized API response, validation outcome, format/constraint flags, and timestamps. An output that has valid JSON structure but exceeds a word or note limit is kept for evaluation and marked with `constraint_violations`. A response wrapped in a code fence or containing literal newlines inside a JSON string is mechanically parsed and kept, but marked as nonconforming; a bare unstructured poem is not guessed into the schema.

Preparation can select a smaller dev subset or a different model/repetition count:

```powershell
python -m src.cli prepare --split dev --limit-poems 2 --repetitions 1 --models gpt-6.1-sol,gpt-6-luna --manifest results/dev_subset_manifest.json
```

## Blind judging

The primary judge uses `gpt-6.1-sol`. It scores test repetitions 0–2 in 144 batches of six. Each poem/repetition group contains the 18 translations from the 3 generation models × 6 conditions, shuffled with a fixed seed and split into batches of six. Judges see the source poem, critical units, and each translation's stable opaque ID, translation, and notes; model IDs, conditions, generation prompts, author, and title are not provided.

```powershell
python -m src.cli judge --variant primary --generation-dir results/generation --generation-manifest results/generation_manifest.json --results-dir results/judging
```

The judge returns six 1–5 integer scores (`meaning`, `imagery`, `tone`, `english`, `faithful_fit`, and `poetic_fit`) and a status, exact output quote, and explanation for every critical unit. Quoted evidence is checked against the translation or its notes. Online checks originally require exact substrings; the final offline acceptance step applies one shared policy to all variants and attempts, normalizing whitespace and case only. Punctuation, words, literal backslash-n text, and scores are unchanged. For each batch, the earliest attempt passing that shared policy is retained, independently of its score. A missing unit may have an empty quote if its explanation identifies what is absent. Notes can inform `faithful_fit` only; they do not compensate for translation omissions in other dimensions or unit coverage.

Use `--limit-batches N` for the first N deterministic batches, or `--sample-batches N --sample-seed 20261025` for a reproducible random subset. Batch size can be reduced to 3 with `--batch-size 3`. The preset secondary judge is `gpt-5.6-luna` on repetition 0 for all 16 test poems (48 batches, 288 translations). The reversed-order sensitivity preset uses the primary judge on repetition 0 for the first four test poem IDs in sorted order (12 batches, 72 translations), reversing each batch's item order while retaining opaque IDs and batch membership:

```powershell
python -m src.cli judge --variant secondary --generation-dir results/generation --generation-manifest results/generation_manifest.json --results-dir results/judging
python -m src.cli judge --variant reversed --generation-dir results/generation --generation-manifest results/generation_manifest.json --results-dir results/judging
```

For example, adding `--split dev --generation-dir results/generation_dev_v2 --generation-manifest results/dev_v2_manifest.json --results-dir results/judging_dev_v2` runs the 12-batch dev primary judge. The primary and secondary plans use the same opaque IDs and batch order for repetition 0; the reversed plan changes only the order inside each selected batch. The secondary and reversed comparisons measure sensitivity to judge identity and within-batch order, respectively. They can reveal patterns consistent with self-preference or position effects but do not identify or eliminate those effects by themselves.

Raw judging attempts, including request messages, opaque mappings, API responses, and validation outcomes, are appended to one JSONL file per judging variant. All raw attempts are retained. `python -m src.revalidate` writes separate derived `results/judging_accepted` files with source attempt numbers and original validation metadata; it never modifies raw responses. The derivation refuses an incomplete experiment or any batch without an acceptable response. An online judge command can exit nonzero when its exact-quote checks reject only whitespace or capitalization differences. Inspect the raw validation status, then run the shared offline acceptance step. It must accept all 144 primary, 48 secondary, and 12 reversed batches; genuine missing content or unsupported quotations are not repaired. Rerunning with the same plan resumes unfinished batches and preserves accepted judgments. Keep different seeds or batch sizes in separate result directories because they define different judge tasks.

## Summary outputs

Summarize after generation and judging. Pass the result locations used above:

```powershell
python -m src.revalidate
python -m src.cli summarize --manifest results/generation_manifest.json --generation-dir results/generation --judge-results-dir results/judging_accepted --output-dir results/summary
```

The command writes:

- `summary.json`: aggregate estimates, poem-cluster bootstrap intervals, sample counts, and aggregation metadata.
- `summary_long.csv`: score means and paired condition differences, plus generation format-validity and word-count metrics, with overall, genre, model, and genre-by-model strata.
- `poem_paired_changes.csv`: per-poem paired condition changes by score dimension.
- `judge_scores_long.csv`: validated scores and critical-unit outcomes mapped back to the task metadata.
- `generation_long.csv`: one row per assigned generation task, including attempt count, schema validity, raw JSON conformance, constraint violations, and translation/note word counts.

After the complete formal test has all 864 distinct primary judgments, build the three figures and their source data:

```powershell
python -m src.plots --summary-dir results/summary --output-dir figures
```

The plotting command checks that `judge_scores_long.csv` contains all 864 primary test evaluations; a dev or partial summary is not sufficient. It writes `figure_data.json` into the summary directory and `quality_dimensions.png`, `context_effects.png`, and `purpose_comparison.png` into the figure output directory. The figures show model-by-condition scores with poem-cluster bootstrap intervals, paired effects of adding semantic notes, and how the two intended-use scores jointly change when the translation purpose is changed.

Scoring first averages repetitions within each poem × generation model × condition, then averages the three models within each poem × condition, then averages across poems. Model and genre strata are also summarized using poems as the sampling unit. The 95% intervals use a poem-cluster bootstrap with seed `20261025` and 2,000 resamples by default. This avoids treating repeated translations of a poem as independent poem observations.

## Interpretation limits

The scores are judgments from language models, not independent human ratings. Critical units make selected semantic requirements auditable but cannot represent every valid interpretation of a poem. Chinese often leaves subjects and relationships implicit; a reasonable English pronoun needed for sentence grammar is not automatically an error, while unsupported specificity or a new event can be. The comparison is limited to the selected poems, generation models, prompt conditions, and local inference setup. The secondary and reversed-order variants are sensitivity checks and do not prove that self-preference or position effects are absent.
