"""Aggregate poem-level scores and generation diagnostics."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_DIR = ROOT / "results" / "summary"
BOOTSTRAP_SEED = 20261025
BOOTSTRAP_SAMPLES = 2000
SCORE_DIMENSIONS = ("meaning", "imagery", "tone", "english", "faithful_fit", "poetic_fit")
CONDITIONS = ("A", "B", "C", "D", "E", "F")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(record)
    return records


def _mean(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _bootstrap_ci(
    values: list[tuple[str, float]],
    *,
    key: str,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float | None, float | None]:
    if len(values) < 2 or samples < 1:
        return None, None
    digest = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed ^ digest)
    observations = [value for _, value in sorted(values)]
    boot_means = []
    for _ in range(samples):
        draw = [observations[rng.randrange(len(observations))] for _ in observations]
        boot_means.append(sum(draw) / len(draw))
    boot_means.sort()
    lower_index = max(0, int(0.025 * samples))
    upper_index = min(samples - 1, int(0.975 * samples))
    return boot_means[lower_index], boot_means[upper_index]


def _score_rows(
    task_map: dict[str, dict[str, Any]],
    judging_results_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    scores: list[dict[str, Any]] = []
    variant_counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    seen_tasks: set[tuple[str, str, str]] = set()
    for variant in ("primary", "secondary", "reversed"):
        for record in _read_jsonl(judging_results_dir / f"{variant}.jsonl"):
            if record.get("variant") != variant or record.get("validation", {}).get("ok") is not True:
                continue
            variant_key = (variant, str(record.get("judge_model")), str(record.get("batch_id")))
            if variant_key in seen:
                raise ValueError(f"multiple accepted judging responses for {variant_key}")
            seen.add(variant_key)
            mapping = record.get("request", {}).get("opaque_mapping", [])
            task_by_opaque = {
                item.get("opaque_id"): item.get("task_id")
                for item in mapping
                if isinstance(item, dict) and isinstance(item.get("opaque_id"), str)
            }
            evaluations = record["validation"].get("evaluations") or []
            for evaluation in evaluations:
                opaque_id = evaluation.get("opaque_id")
                task_id = task_by_opaque.get(opaque_id)
                if task_id not in task_map:
                    continue
                task = task_map[task_id]
                score_key = (variant, str(record.get("judge_model")), task_id)
                if score_key in seen_tasks:
                    raise ValueError(f"multiple accepted judge evaluations for task {score_key}")
                seen_tasks.add(score_key)
                row = {
                    "judge_variant": variant,
                    "judge_model": record.get("judge_model"),
                    "judge_batch_id": record.get("batch_id"),
                    "judge_position": next((i for i, item in enumerate(mapping) if item.get("opaque_id") == opaque_id), None),
                    "opaque_id": opaque_id,
                    "task_id": task_id,
                    "poem_id": task["poem_id"],
                    "split": task["split"],
                    "genre": task["genre"],
                    "model": task["model"],
                    "condition": task["condition"],
                    "repetition": task["repetition"],
                    "critical_units_json": json.dumps(evaluation.get("critical_units", []), ensure_ascii=False, separators=(",", ":")),
                }
                row.update({dimension: evaluation["scores"][dimension] for dimension in SCORE_DIMENSIONS})
                scores.append(row)
                variant_counts[variant] += 1
    return scores, dict(variant_counts)


def _generation_rows(manifest: dict[str, Any], generation_results_dir: Path) -> list[dict[str, Any]]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("generation manifest must contain a task list")
    task_map = {task["task_id"]: task for task in tasks}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    splits = sorted({task.get("split") for task in tasks})
    for split in splits:
        for record in _read_jsonl(generation_results_dir / f"{split}.jsonl"):
            task_id = record.get("task_id")
            if task_id in task_map:
                attempts[task_id].append(record)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_attempts = sorted(attempts.get(task["task_id"], []), key=lambda record: record.get("attempt", 0))
        successful = [record for record in task_attempts if record.get("validation", {}).get("ok") is True]
        if len(successful) > 1:
            raise ValueError(f"multiple accepted generation responses for task {task['task_id']}")
        accepted = successful[0] if successful else None
        validation = accepted.get("validation", {}) if accepted else {}
        constraint_violations = validation.get("constraint_violations", [])
        rows.append({
            **task,
            "attempt_count": len(task_attempts),
            "schema_valid": accepted is not None,
            "raw_json_conformant": validation.get("raw_json_conformant") is True if accepted else False,
            "constraint_compliant": accepted is not None and not constraint_violations,
            "constraint_violation_count": len(constraint_violations),
            "constraint_violations": ";".join(constraint_violations),
            "format_violations": ";".join(validation.get("format_violations", [])),
            "translation_words": validation.get("translation_word_count") if accepted else None,
            "notes_words": validation.get("notes_word_count") if accepted else None,
            "notes_count": len(validation.get("notes", [])) if accepted else None,
        })
    return rows


def _poem_model_condition_means(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for key, group in grouped.items():
        base = {name: value for name, value in zip(keys, key)}
        base["task_count"] = len(group)
        base["repetition_count"] = len({row.get("repetition") for row in group})
        for metric in metrics:
            base[metric] = _mean(row.get(metric) for row in group)
        result.append(base)
    return result


def _generation_summary_rows(
    task_rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    metrics = (
        "schema_valid_rate", "raw_json_conformant_rate", "constraint_compliant_rate",
        "translation_words_mean", "notes_words_mean", "notes_count_mean", "attempts_mean",
    )
    value_fields = {
        "schema_valid_rate": "schema_valid",
        "raw_json_conformant_rate": "raw_json_conformant",
        "constraint_compliant_rate": "constraint_compliant",
        "translation_words_mean": "translation_words",
        "notes_words_mean": "notes_words",
        "notes_count_mean": "notes_count",
        "attempts_mean": "attempt_count",
    }
    converted = []
    for row in task_rows:
        clone = dict(row)
        for metric, field in value_fields.items():
            if metric.endswith("_rate"):
                clone[metric] = 1.0 if row[field] else 0.0
            else:
                clone[metric] = row.get(field)
        converted.append(clone)
    model_means = _poem_model_condition_means(
        converted,
        metrics,
        ("split", "poem_id", "genre", "model", "condition"),
    )
    poem_means: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in model_means:
        grouped[(row["split"], row["poem_id"], row["genre"], row["condition"])].append(row)
    for key, group in grouped.items():
        split, poem_id, genre, condition = key
        base = {"split": split, "poem_id": poem_id, "genre": genre, "condition": condition}
        base["model_count"] = len(group)
        for metric in metrics:
            base[metric] = _mean(row.get(metric) for row in group)
        poem_means.append(base)

    output: list[dict[str, Any]] = []

    def emit(group_type: str, group_label: str, metric: str, condition: str, values: list[tuple[str, float]], *, split: str, genre: str | None = None, model: str | None = None) -> None:
        ci_low, ci_high = _bootstrap_ci(values, key=f"gen|{split}|{group_type}|{group_label}|{metric}|{condition}", seed=seed, samples=bootstrap_samples)
        output.append({
            "row_type": "generation_metric_mean",
            "judge_variant": "",
            "judge_model": "",
            "split": split,
            "metric": metric,
            "dimension": "",
            "condition": condition,
            "condition_a": "",
            "condition_b": "",
            "group_type": group_type,
            "group_label": group_label,
            "poem_id": "",
            "genre": genre or "",
            "model": model or "",
            "n_poems": len(values),
            "mean": _mean(value for _, value in values),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_seed": seed,
        })

    splits = sorted({row["split"] for row in poem_means})
    for split in splits:
        relevant = [row for row in poem_means if row["split"] == split]
        for condition in CONDITIONS:
            for metric in metrics:
                values = [(row["poem_id"], row[metric]) for row in relevant if row["condition"] == condition and row.get(metric) is not None]
                if values:
                    emit("overall", "all", metric, condition, values, split=split)
                for genre in sorted({row["genre"] for row in relevant}):
                    vals = [(row["poem_id"], row[metric]) for row in relevant if row["genre"] == genre and row["condition"] == condition and row.get(metric) is not None]
                    if vals:
                        emit("genre", genre, metric, condition, vals, split=split, genre=genre)
                relevant_model = [row for row in model_means if row["split"] == split and row["condition"] == condition]
                for model in sorted({row["model"] for row in relevant_model}):
                    vals = [(row["poem_id"], row[metric]) for row in relevant_model if row["model"] == model and row.get(metric) is not None]
                    if vals:
                        emit("model", model, metric, condition, vals, split=split, model=model)
                    for genre in sorted({row["genre"] for row in relevant_model if row["model"] == model}):
                        vals = [(row["poem_id"], row[metric]) for row in relevant_model if row["model"] == model and row["genre"] == genre and row.get(metric) is not None]
                        if vals:
                            emit("genre_model", f"{genre}|{model}", metric, condition, vals, split=split, genre=genre, model=model)
    return output


def _score_summary_rows(
    score_rows: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        key = (row["judge_variant"], row["judge_model"], row["split"], row["poem_id"], row["genre"], row["model"], row["condition"])
        model_groups[key].append(row)
    poem_model_means = []
    for key, group in model_groups.items():
        variant, judge_model, split, poem_id, genre, model, condition = key
        result = {
            "judge_variant": variant,
            "judge_model": judge_model,
            "split": split,
            "poem_id": poem_id,
            "genre": genre,
            "model": model,
            "condition": condition,
            "repetition_count": len({row["repetition"] for row in group}),
        }
        for dimension in SCORE_DIMENSIONS:
            result[dimension] = _mean(row[dimension] for row in group)
        by_repetition: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_repetition[row["repetition"]].append(row)
        result["repetition_scores"] = {
            dimension: {
                str(repetition): _mean(row[dimension] for row in repetition_group)
                for repetition, repetition_group in by_repetition.items()
            }
            for dimension in SCORE_DIMENSIONS
        }
        poem_model_means.append(result)

    poem_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in poem_model_means:
        key = (row["judge_variant"], row["judge_model"], row["split"], row["poem_id"], row["genre"], row["condition"])
        poem_groups[key].append(row)
    poem_condition_means = []
    for key, group in poem_groups.items():
        variant, judge_model, split, poem_id, genre, condition = key
        result = {
            "judge_variant": variant,
            "judge_model": judge_model,
            "split": split,
            "poem_id": poem_id,
            "genre": genre,
            "condition": condition,
            "model_count": len({row["model"] for row in group}),
        }
        for dimension in SCORE_DIMENSIONS:
            result[dimension] = _mean(row[dimension] for row in group)
        poem_condition_means.append(result)

    output: list[dict[str, Any]] = []

    def emit_mean(
        group_type: str,
        group_label: str,
        dimension: str,
        condition: str,
        values: list[tuple[str, float]],
        *,
        variant: str,
        judge_model: str,
        split: str,
        genre: str | None = None,
        model: str | None = None,
    ) -> None:
        ci_low, ci_high = _bootstrap_ci(values, key=f"score|{variant}|{judge_model}|{split}|{group_type}|{group_label}|{dimension}|{condition}", seed=seed, samples=bootstrap_samples)
        output.append({
            "row_type": "dimension_mean",
            "judge_variant": variant,
            "judge_model": judge_model,
            "split": split,
            "metric": "score",
            "dimension": dimension,
            "condition": condition,
            "condition_a": "",
            "condition_b": "",
            "group_type": group_type,
            "group_label": group_label,
            "poem_id": "",
            "genre": genre or "",
            "model": model or "",
            "n_poems": len(values),
            "mean": _mean(value for _, value in values),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_seed": seed,
        })

    contexts = sorted({(row["judge_variant"], row["judge_model"], row["split"]) for row in poem_condition_means})
    for variant, judge_model, split in contexts:
        relevant_poem = [row for row in poem_condition_means if (row["judge_variant"], row["judge_model"], row["split"]) == (variant, judge_model, split)]
        relevant_model = [row for row in poem_model_means if (row["judge_variant"], row["judge_model"], row["split"]) == (variant, judge_model, split)]
        for condition in CONDITIONS:
            for dimension in SCORE_DIMENSIONS:
                vals = [(row["poem_id"], row[dimension]) for row in relevant_poem if row["condition"] == condition and row.get(dimension) is not None]
                if vals:
                    emit_mean("overall", "all", dimension, condition, vals, variant=variant, judge_model=judge_model, split=split)
                for genre in sorted({row["genre"] for row in relevant_poem if row["condition"] == condition}):
                    vals = [(row["poem_id"], row[dimension]) for row in relevant_poem if row["genre"] == genre and row["condition"] == condition and row.get(dimension) is not None]
                    if vals:
                        emit_mean("genre", genre, dimension, condition, vals, variant=variant, judge_model=judge_model, split=split, genre=genre)
                for model in sorted({row["model"] for row in relevant_model if row["condition"] == condition}):
                    vals = [(row["poem_id"], row[dimension]) for row in relevant_model if row["model"] == model and row["condition"] == condition and row.get(dimension) is not None]
                    if vals:
                        emit_mean("model", model, dimension, condition, vals, variant=variant, judge_model=judge_model, split=split, model=model)
                    for genre in sorted({row["genre"] for row in relevant_model if row["model"] == model and row["condition"] == condition}):
                        vals = [(row["poem_id"], row[dimension]) for row in relevant_model if row["model"] == model and row["genre"] == genre and row["condition"] == condition and row.get(dimension) is not None]
                        if vals:
                            emit_mean("genre_model", f"{genre}|{model}", dimension, condition, vals, variant=variant, judge_model=judge_model, split=split, genre=genre, model=model)

    paired_rows: list[dict[str, Any]] = []
    for variant, judge_model, split in contexts:
        relevant = [row for row in poem_model_means if (row["judge_variant"], row["judge_model"], row["split"]) == (variant, judge_model, split)]
        by_poem: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
        for row in relevant:
            by_poem[row["poem_id"]][row["condition"]][row["model"]] = row
        for poem_id, conditions in sorted(by_poem.items()):
            genre = next((row["genre"] for model_rows in conditions.values() for row in model_rows.values()), "")
            for condition_a, condition_b in itertools.combinations(sorted(conditions), 2):
                common_models = sorted(set(conditions[condition_a]) & set(conditions[condition_b]))
                for dimension in SCORE_DIMENSIONS:
                    model_deltas: list[float] = []
                    for model in common_models:
                        row_a = conditions[condition_a][model]
                        row_b = conditions[condition_b][model]
                        reps_a = row_a["repetition_scores"][dimension]
                        reps_b = row_b["repetition_scores"][dimension]
                        common_repetitions = sorted(set(reps_a) & set(reps_b), key=int)
                        if not common_repetitions:
                            continue
                        value_a = _mean(reps_a[repetition] for repetition in common_repetitions)
                        value_b = _mean(reps_b[repetition] for repetition in common_repetitions)
                        model_deltas.append(value_b - value_a)
                    if not model_deltas:
                        continue
                    paired_rows.append({
                        "judge_variant": variant,
                        "judge_model": judge_model,
                        "split": split,
                        "poem_id": poem_id,
                        "genre": genre,
                        "dimension": dimension,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "delta_b_minus_a": _mean(model_deltas),
                    })
    paired_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        paired_groups[(row["judge_variant"], row["judge_model"], row["split"], row["dimension"], row["condition_a"], row["condition_b"])].append(row)
    for key, group in paired_groups.items():
        variant, judge_model, split, dimension, condition_a, condition_b = key
        vals = [(row["poem_id"], row["delta_b_minus_a"]) for row in group]
        ci_low, ci_high = _bootstrap_ci(vals, key=f"delta|{variant}|{judge_model}|{split}|{dimension}|{condition_a}|{condition_b}", seed=seed, samples=bootstrap_samples)
        output.append({
            "row_type": "paired_condition_change_mean",
            "judge_variant": variant,
            "judge_model": judge_model,
            "split": split,
            "metric": "paired_score_difference",
            "dimension": dimension,
            "condition": "",
            "condition_a": condition_a,
            "condition_b": condition_b,
            "group_type": "overall",
            "group_label": "all",
            "poem_id": "",
            "genre": "",
            "model": "",
            "n_poems": len(vals),
            "mean": _mean(value for _, value in vals),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_seed": seed,
        })
        for genre in sorted({row["genre"] for row in group}):
            genre_vals = [(row["poem_id"], row["delta_b_minus_a"]) for row in group if row["genre"] == genre]
            low, high = _bootstrap_ci(genre_vals, key=f"delta|{variant}|{judge_model}|{split}|{genre}|{dimension}|{condition_a}|{condition_b}", seed=seed, samples=bootstrap_samples)
            output.append({
                "row_type": "paired_condition_change_mean",
                "judge_variant": variant,
                "judge_model": judge_model,
                "split": split,
                "metric": "paired_score_difference",
                "dimension": dimension,
                "condition": "",
                "condition_a": condition_a,
                "condition_b": condition_b,
                "group_type": "genre",
                "group_label": genre,
                "poem_id": "",
                "genre": genre,
                "model": "",
                "n_poems": len(genre_vals),
                "mean": _mean(value for _, value in genre_vals),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_seed": seed,
            })
    return output, paired_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(
    manifest_path: Path,
    generation_results_dir: Path,
    judging_results_dir: Path,
    output_dir: Path = DEFAULT_SUMMARY_DIR,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("generation manifest must contain at least one task")
    task_map = {task["task_id"]: task for task in tasks}
    generation_rows = _generation_rows(manifest, generation_results_dir)
    score_rows, judge_counts = _score_rows(task_map, judging_results_dir)
    summary_rows = _generation_summary_rows(
        generation_rows,
        seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    score_summary, paired_rows = _score_summary_rows(
        score_rows,
        seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    summary_rows.extend(score_summary)
    for row in paired_rows:
        summary_rows.append({
            "row_type": "paired_poem_change",
            "judge_variant": row["judge_variant"],
            "judge_model": row["judge_model"],
            "split": row["split"],
            "metric": "paired_score_difference",
            "dimension": row["dimension"],
            "condition": "",
            "condition_a": row["condition_a"],
            "condition_b": row["condition_b"],
            "group_type": "poem",
            "group_label": row["poem_id"],
            "poem_id": row["poem_id"],
            "genre": row["genre"],
            "model": "",
            "n_poems": 1,
            "mean": row["delta_b_minus_a"],
            "ci_low": None,
            "ci_high": None,
            "bootstrap_seed": bootstrap_seed,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    generation_columns = [
        "task_id", "split", "poem_id", "genre", "model", "condition", "repetition",
        "attempt_count", "schema_valid", "raw_json_conformant", "constraint_compliant",
        "constraint_violation_count", "constraint_violations", "format_violations",
        "translation_words", "notes_words", "notes_count",
    ]
    score_columns = [
        "judge_variant", "judge_model", "judge_batch_id", "judge_position", "opaque_id", "task_id",
        "poem_id", "split", "genre", "model", "condition", "repetition", *SCORE_DIMENSIONS,
        "critical_units_json",
    ]
    summary_columns = [
        "row_type", "judge_variant", "judge_model", "split", "metric", "dimension", "condition",
        "condition_a", "condition_b", "group_type", "group_label", "poem_id", "genre", "model",
        "n_poems", "mean", "ci_low", "ci_high", "bootstrap_seed",
    ]
    paired_columns = [
        "judge_variant", "judge_model", "split", "poem_id", "genre", "dimension",
        "condition_a", "condition_b", "delta_b_minus_a",
    ]
    _write_csv(output_dir / "generation_long.csv", generation_rows, generation_columns)
    _write_csv(output_dir / "judge_scores_long.csv", score_rows, score_columns)
    _write_csv(output_dir / "summary_long.csv", summary_rows, summary_columns)
    _write_csv(output_dir / "poem_paired_changes.csv", paired_rows, paired_columns)

    summary_json = {
        "summary_version": 1,
        "generation_manifest": manifest_path.name,
        "expected_generation_tasks": len(tasks),
        "successful_generation_tasks": sum(row["schema_valid"] for row in generation_rows),
        "judge_evaluations_by_variant": judge_counts,
        "bootstrap": {
            "unit": "poem",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "interval": "percentile 95%",
        },
        "aggregation": "average repetitions within poem × model × condition, average models within poem × condition, then average poems; model and genre strata use poems as the resampling unit",
        "mean_rows": summary_rows,
        "paired_poem_changes": paired_rows,
        "artifacts": [
            "generation_long.csv",
            "judge_scores_long.csv",
            "summary_long.csv",
            "poem_paired_changes.csv",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "expected_generation_tasks": len(tasks),
        "successful_generation_tasks": sum(row["schema_valid"] for row in generation_rows),
        "judge_score_rows": len(score_rows),
        "summary_rows": len(summary_rows),
        "paired_poem_rows": len(paired_rows),
        "output_dir": str(output_dir),
    }
