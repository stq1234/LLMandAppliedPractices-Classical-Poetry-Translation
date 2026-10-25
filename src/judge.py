"""Blind, batched evaluation of generated translations."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Any, Callable

from . import api_client
from .generate import (
    CONDITIONS,
    DATA_PATH,
    DEFAULT_RESULTS_DIR,
    MODELS,
    SHUFFLE_SEED,
    _task_id,
    _validate_poems,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_RESULTS_DIR = ROOT / "results" / "judging"
PRIMARY_JUDGE_MODEL = "gpt-6.1-sol"
SECONDARY_JUDGE_MODEL = "gpt-5.6-luna"
BATCH_SIZE = 6
MAX_COMPLETION_TOKENS = 8192
REASONING_EFFORT = "low"
DIMENSIONS = (
    "meaning", "imagery", "tone", "english", "faithful_fit", "poetic_fit",
)
UNIT_STATUSES = {"present", "partial", "missing", "wrong"}
QUOTE_MATCH_POLICY = "whitespace_collapse_casefold_substring_v1"

JUDGE_INSTRUCTIONS = """Evaluate each English poem translation against the supplied Chinese source. Score each translation on six separate dimensions from 1 to 5 using integers only. The common anchors are: 1 = severe failure; 3 = broadly understandable or basically present but with clear defects; 5 = accurate or excellent for that dimension. Scores 2 and 4 are intermediate.

Dimensions: meaning = semantic accuracy and coverage; imagery = preservation and clarity of source images; tone = preservation of register and emotional direction; english = grammar, idiom, and fluency; faithful_fit = suitability as a faithful literary translation for an English-only reader; poetic_fit = effectiveness as an English poem, including rhythm and lineation. Consider the translation as the primary work. Notes may help only with faithful_fit; they must not compensate for material absent from the translation in the other dimensions or in critical-unit coverage. Do not reward length, the number of notes, or wording merely repeated between translation and notes. Do not infer which prompt, condition, or model produced an item.

English syntax may require an explicit subject or pronoun. A reasonable I/we/she or similar subject added to form a complete English sentence is not automatically an error. Penalize only unsupported specificity about identity, relationship, or a new event, and allow ambiguity where the Chinese source permits it.

For every critical unit, assign one status based on the translation: present (meaning accurately conveyed), partial (only some meaning conveyed), missing (not conveyed), or wrong (materially distorted). Give a short exact quote from the translation as evidence when possible; a note may be quoted only when it directly clarifies that unit. For a missing unit with no relevant excerpt, quote may be empty, but explanation must say what is absent. Every non-empty quote must be copied exactly from that item's translation or notes. Keep explanations concise.

Return exactly one JSON object, with no prose or code fence, in this schema:
{"evaluations":[{"opaque_id":"...","scores":{"meaning":1,"imagery":1,"tone":1,"english":1,"faithful_fit":1,"poetic_fit":1},"critical_units":[{"id":"...","status":"present","quote":"...","explanation":"..."}]}]}
Include every supplied opaque_id exactly once. Include every supplied critical-unit id exactly once for each translation. Do not add dimensions, units, or items."""


def _opaque_id(task_id: str, seed: int = SHUFFLE_SEED) -> str:
    digest = hashlib.sha256(f"{seed}\0{task_id}".encode("utf-8")).hexdigest()[:14]
    return f"x{digest}"


def _load_generation_records(results_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    path = results_dir / f"{split}.jsonl"
    if not path.exists():
        return accepted
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if record.get("split") != split:
                continue
            if record.get("validation", {}).get("ok") is not True:
                continue
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError(f"missing task_id at {path}:{line_number}")
            if task_id in accepted:
                raise ValueError(f"multiple accepted generation responses found for task {task_id}")
            accepted[task_id] = record
    return accepted


def _expected_task(poem: dict[str, Any], model: str, condition: str, repetition: int) -> dict[str, Any]:
    return {
        "task_id": _task_id(poem["id"], condition, model, repetition),
        "poem_id": poem["id"],
        "split": poem["split"],
        "genre": poem["genre"],
        "model": model,
        "condition": condition,
        "repetition": repetition,
    }


def _batch_order(poem_id: str, repetition: int, items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: item["task_id"])
    group_seed = int(hashlib.sha256(f"{seed}\0{poem_id}\0{repetition}".encode("utf-8")).hexdigest()[:16], 16)
    random.Random(group_seed).shuffle(ordered)
    return ordered


def build_judge_plan(
    *,
    variant: str,
    split: str = "test",
    generation_results_dir: Path = DEFAULT_RESULTS_DIR,
    generation_manifest_path: Path | None = None,
    data_path: Path = DATA_PATH,
    judge_model: str | None = None,
    batch_size: int = BATCH_SIZE,
    seed: int = SHUFFLE_SEED,
) -> dict[str, Any]:
    if variant not in {"primary", "secondary", "reversed"}:
        raise ValueError("variant must be primary, secondary, or reversed")
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if batch_size not in {3, 6}:
        raise ValueError("batch_size must be 3 or 6")
    default_model = SECONDARY_JUDGE_MODEL if variant == "secondary" else PRIMARY_JUDGE_MODEL
    selected_judge = judge_model or default_model
    if selected_judge not in MODELS:
        raise ValueError(f"judge_model must be one of: {', '.join(MODELS)}")

    with data_path.open("r", encoding="utf-8") as stream:
        poems = _validate_poems(json.load(stream), require_full_dataset=True)
    selected_poems = sorted((p for p in poems if p["split"] == split), key=lambda poem: poem["id"])
    manifest_repetitions: tuple[int, ...] = (0,) if split == "dev" else (0, 1, 2)
    if generation_manifest_path is not None:
        with generation_manifest_path.open("r", encoding="utf-8") as stream:
            generation_manifest = json.load(stream)
        if generation_manifest.get("manifest_version") != 1 or not isinstance(generation_manifest.get("tasks"), list):
            raise ValueError("generation manifest is invalid or unsupported")
        generation_spec = generation_manifest.get("generation", {})
        if set(generation_spec.get("models", [])) != set(MODELS):
            raise ValueError("judge requires all three configured generation models")
        if set(generation_spec.get("conditions", [])) != set(CONDITIONS):
            raise ValueError("judge requires all six configured generation conditions")
        scoped_tasks = [task for task in generation_manifest["tasks"] if task.get("split") == split]
        if not scoped_tasks:
            raise ValueError(f"generation manifest has no tasks for split {split}")
        selected_ids = {task.get("poem_id") for task in scoped_tasks}
        selected_poems = [poem for poem in selected_poems if poem["id"] in selected_ids]
        manifest_repetitions = tuple(sorted({task.get("repetition") for task in scoped_tasks}))
        if not selected_poems or not manifest_repetitions or any(type(value) is not int for value in manifest_repetitions):
            raise ValueError("generation manifest has an invalid poem or repetition scope")
        expected_factor = {(poem_id, model, condition, repetition)
                           for poem_id in selected_ids
                           for model in MODELS
                           for condition in CONDITIONS
                           for repetition in manifest_repetitions}
        observed_factor = {(task.get("poem_id"), task.get("model"), task.get("condition"), task.get("repetition"))
                           for task in scoped_tasks}
        if observed_factor != expected_factor or len(scoped_tasks) != len(expected_factor):
            raise ValueError("generation manifest is not a complete 3-model by 6-condition task grid")
    default_repetitions = manifest_repetitions
    if variant == "reversed":
        selected_poems = selected_poems[:4]
        repetitions = (0,)
    elif variant == "secondary":
        repetitions = (0,)
    else:
        repetitions = default_repetitions
    successes = _load_generation_records(generation_results_dir, split)
    plan_batches: list[dict[str, Any]] = []

    for poem in selected_poems:
        for repetition in repetitions:
            group = [
                _expected_task(poem, model, condition, repetition)
                for model in MODELS
                for condition in CONDITIONS
            ]
            ordered = _batch_order(poem["id"], repetition, group, seed)
            for batch_index in range(0, len(ordered), batch_size):
                batch_items: list[dict[str, Any]] = []
                for task in ordered[batch_index : batch_index + batch_size]:
                    generated = successes.get(task["task_id"])
                    valid = generated is not None
                    batch_items.append({
                        **task,
                        "opaque_id": _opaque_id(task["task_id"], seed),
                        "available": valid,
                        "translation": generated["validation"]["translation"] if valid else None,
                        "notes": generated["validation"]["notes"] if valid else None,
                    })
                batch_number = batch_index // batch_size
                if variant == "reversed":
                    batch_items.reverse()
                plan_batches.append({
                    "batch_id": f"{poem['id']}__r{repetition}__b{batch_number:02d}",
                    "split": split,
                    "poem_id": poem["id"],
                    "genre": poem["genre"],
                    "repetition": repetition,
                    "batch_number": batch_number,
                    "items": batch_items,
                    "available": all(item["available"] for item in batch_items),
                })
    return {
        "judge_plan_version": 1,
        "variant": variant,
        "split": split,
        "generation_manifest": generation_manifest_path.name if generation_manifest_path is not None else None,
        "judge_model": selected_judge,
        "judge_instructions": JUDGE_INSTRUCTIONS,
        "seed": seed,
        "batch_size": batch_size,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": "omitted",
        "top_p": "omitted",
        "source_poems": {
            poem["id"]: {
                "text": poem["text"],
                "critical_units": poem["critical_units"],
            }
            for poem in selected_poems
        },
        "batches": plan_batches,
    }


def build_judge_messages(
    poem: dict[str, Any],
    batch: dict[str, Any],
    judge_instructions: str = JUDGE_INSTRUCTIONS,
) -> list[dict[str, str]]:
    visible_items = [
        {"opaque_id": item["opaque_id"], "translation": item["translation"], "notes": item["notes"]}
        for item in batch["items"]
    ]
    user_payload = {
        "source_poem_zh": poem["text"],
        "critical_units": poem["critical_units"],
        "translations_to_score": visible_items,
    }
    return [
        {"role": "system", "content": judge_instructions},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def _quote_is_substring(quote: str, text: str, *, normalize_quotes: bool) -> bool:
    if not normalize_quotes:
        return quote in text
    normalized_quote = " ".join(quote.split()).casefold()
    normalized_text = " ".join(text.split()).casefold()
    return bool(normalized_quote) and normalized_quote in normalized_text


def parse_judgment(
    content: Any,
    batch: dict[str, Any],
    *,
    normalize_quotes: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(content, str):
        return None, ["response content is not text"]
    body = content.strip()
    if body.startswith("```") and body.endswith("```"):
        first_newline = body.find("\n")
        if first_newline < 0:
            return None, ["code fence has no JSON body"]
        body = body[first_newline + 1 : -3].strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc.msg} at character {exc.pos}"]
    if not isinstance(parsed, dict) or set(parsed) != {"evaluations"} or not isinstance(parsed["evaluations"], list):
        return None, ["top-level JSON must contain only an evaluations array"]
    errors: list[str] = []
    expected_items = {item["opaque_id"]: item for item in batch["items"]}
    expected_ids = set(expected_items)
    seen_ids: set[str] = set()
    if len(parsed["evaluations"]) != len(expected_ids):
        errors.append("evaluations array length does not match the batch")
    source_units = {unit["id"] for unit in batch["critical_units"]}
    for index, evaluation in enumerate(parsed["evaluations"]):
        if not isinstance(evaluation, dict):
            errors.append(f"evaluation {index} must be an object")
            continue
        opaque_id = evaluation.get("opaque_id")
        if not isinstance(opaque_id, str) or opaque_id not in expected_ids:
            errors.append(f"evaluation {index} has an unknown opaque_id")
            continue
        if opaque_id in seen_ids:
            errors.append(f"opaque_id {opaque_id} appears more than once")
        seen_ids.add(opaque_id)
        if set(evaluation) != {"opaque_id", "scores", "critical_units"}:
            errors.append(f"evaluation {opaque_id} has unexpected or missing keys")
        scores = evaluation.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
            errors.append(f"evaluation {opaque_id} must contain exactly six score dimensions")
        elif any(type(scores[dimension]) is not int or not 1 <= scores[dimension] <= 5 for dimension in DIMENSIONS):
            errors.append(f"evaluation {opaque_id} scores must be integers from 1 to 5")
        units = evaluation.get("critical_units")
        if not isinstance(units, list):
            errors.append(f"evaluation {opaque_id} critical_units must be a list")
            continue
        got_units: set[str] = set()
        item = expected_items[opaque_id]
        available_quotes = [item["translation"], *item["notes"]]
        for unit_index, unit in enumerate(units):
            if not isinstance(unit, dict) or set(unit) != {"id", "status", "quote", "explanation"}:
                errors.append(f"evaluation {opaque_id} unit {unit_index} has an invalid shape")
                continue
            unit_id = unit.get("id")
            if not isinstance(unit_id, str) or unit_id not in source_units:
                errors.append(f"evaluation {opaque_id} unit {unit_index} has an unknown id")
                continue
            if unit_id in got_units:
                errors.append(f"evaluation {opaque_id} repeats critical unit {unit_id}")
            got_units.add(unit_id)
            if unit.get("status") not in UNIT_STATUSES:
                errors.append(f"evaluation {opaque_id} unit {unit_id} has an invalid status")
            quote = unit.get("quote")
            explanation = unit.get("explanation")
            if not isinstance(quote, str) or not isinstance(explanation, str) or not explanation.strip():
                errors.append(f"evaluation {opaque_id} unit {unit_id} needs string quote and non-empty explanation")
                continue
            status = unit.get("status")
            if status != "missing" and not quote.strip():
                errors.append(f"evaluation {opaque_id} unit {unit_id} needs evidence for status {status}")
            if quote and not any(
                _quote_is_substring(quote, text, normalize_quotes=normalize_quotes)
                for text in available_quotes
            ):
                errors.append(f"evaluation {opaque_id} unit {unit_id} quote is not an exact translation/note substring")
        if got_units != source_units:
            errors.append(f"evaluation {opaque_id} does not cover every critical unit")
    if seen_ids != expected_ids:
        errors.append("evaluations do not contain every expected opaque_id exactly once")
    return (parsed if not errors else None), errors


def _load_attempts(path: Path) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return attempts
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            attempts.setdefault(record["judge_task_id"], []).append(record)
    return attempts


def _judge_record(
    plan: dict[str, Any],
    batch: dict[str, Any],
    attempt_number: int,
    messages: list[dict[str, str]],
    response: dict[str, Any],
) -> dict[str, Any]:
    parsed, errors = parse_judgment(response.get("content"), {**batch, "critical_units": plan["source_poems"][batch["poem_id"]]["critical_units"]}) if response.get("status") == "complete" else (None, [])
    return {
        "judge_task_id": f"{plan['variant']}:{plan['judge_model']}:{batch['batch_id']}",
        "variant": plan["variant"],
        "judge_model": plan["judge_model"],
        "batch_id": batch["batch_id"],
        "poem_id": batch["poem_id"],
        "split": batch["split"],
        "genre": batch["genre"],
        "repetition": batch["repetition"],
        "attempt": attempt_number,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request": {
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": "omitted",
            "top_p": "omitted",
            "opaque_mapping": [
                {"opaque_id": item["opaque_id"], "task_id": item["task_id"], "model": item["model"], "condition": item["condition"]}
                for item in batch["items"]
            ],
        },
        "api_response": response,
        "validation": {"ok": parsed is not None, "errors": errors, "evaluations": parsed["evaluations"] if parsed else None},
    }


def run_judging(
    plan_path: Path,
    results_dir: Path = DEFAULT_JUDGE_RESULTS_DIR,
    *,
    limit_batches: int | None = None,
    sample_batches: int | None = None,
    sample_seed: int = SHUFFLE_SEED,
    max_retries: int = 2,
    workers: int = 1,
    request_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, int]:
    if limit_batches is not None and sample_batches is not None:
        raise ValueError("choose either limit_batches or sample_batches")
    if limit_batches is not None and limit_batches < 1:
        raise ValueError("limit_batches must be positive")
    if sample_batches is not None and sample_batches < 1:
        raise ValueError("sample_batches must be positive")
    if not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be between 0 and 2")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    with plan_path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("judge_plan_version") != 1:
        raise ValueError("unsupported judge plan version")
    if not isinstance(plan.get("judge_instructions"), str):
        raise ValueError("judge plan must freeze the full judge prompt")
    batches = list(plan.get("batches", []))
    if limit_batches is not None:
        batches = batches[:limit_batches]
    elif sample_batches is not None:
        if sample_batches > len(batches):
            raise ValueError(f"sample_batches {sample_batches} exceeds available batch count {len(batches)}")
        indices = sorted(random.Random(sample_seed).sample(range(len(batches)), sample_batches))
        batches = [batches[index] for index in indices]
    variant_path = results_dir / f"{plan['variant']}.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    attempts_by_task = _load_attempts(variant_path)
    append_lock = threading.Lock()
    counts = {"selected": len(batches), "complete": 0, "failed": 0, "skipped": 0, "unavailable": 0}
    count_lock = threading.Lock()
    expected_requests: dict[str, dict[str, Any]] = {}
    for batch in batches:
        poem = plan["source_poems"][batch["poem_id"]]
        prompt_batch = {**batch, "critical_units": poem["critical_units"]}
        messages = build_judge_messages(poem, prompt_batch, plan["judge_instructions"])
        task_id = f"{plan['variant']}:{plan['judge_model']}:{batch['batch_id']}"
        expected_request = {
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": "omitted",
            "top_p": "omitted",
            "opaque_mapping": [
                {"opaque_id": item["opaque_id"], "task_id": item["task_id"], "model": item["model"], "condition": item["condition"]}
                for item in batch["items"]
            ],
        }
        expected_requests[task_id] = expected_request
        for old_record in attempts_by_task.get(task_id, []):
            if old_record.get("request") != expected_request:
                raise ValueError(f"stored judge prompt or parameters differ for {task_id}; refusing to mix runs")
    if request_fn is None:
        api_client.validate_environment()
        request_fn = api_client.request

    def process(batch: dict[str, Any]) -> str:
        if not batch.get("available"):
            return "unavailable"
        task_id = f"{plan['variant']}:{plan['judge_model']}:{batch['batch_id']}"
        previous = attempts_by_task.get(task_id, [])
        if any(item.get("validation", {}).get("ok") is True for item in previous):
            return "skipped"
        if previous and (len(previous) > max_retries or not _retryable_judge(previous[-1])):
            return "failed"
        poem = plan["source_poems"][batch["poem_id"]]
        batch_for_prompt = {**batch, "critical_units": poem["critical_units"]}
        messages = build_judge_messages(poem, batch_for_prompt, plan["judge_instructions"])
        for attempt_number in range(len(previous) + 1, max_retries + 2):
            response = request_fn(
                plan["judge_model"],
                messages,
                max_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort=REASONING_EFFORT,
            )
            record = _judge_record(plan, batch, attempt_number, messages, response)
            _append_judge_jsonl(variant_path, record, append_lock)
            previous.append(record)
            if record["validation"]["ok"]:
                return "complete"
            if attempt_number > max_retries or not _retryable_judge(record):
                return "failed"
        return "failed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            with count_lock:
                counts[outcome] += 1
    return counts


def _retryable_judge(record: dict[str, Any]) -> bool:
    response = record.get("api_response", {})
    if response.get("status") == "truncated":
        return True
    if response.get("status") == "error":
        return response.get("error_type") in {
            "api_error", "timeout", "transport_error", "invalid_json", "invalid_response",
            "non_text_content", "unexpected_finish",
        }
    return response.get("status") == "complete" and record.get("validation", {}).get("ok") is False


def _append_judge_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
