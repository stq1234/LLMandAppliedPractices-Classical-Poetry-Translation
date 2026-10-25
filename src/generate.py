"""Prepare and run the translation generation experiment."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import random
import re
import threading
from pathlib import Path
from typing import Any, Callable

from . import api_client


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "poems.json"
DEFAULT_MANIFEST = ROOT / "results" / "generation_manifest.json"
DEFAULT_RESULTS_DIR = ROOT / "results" / "generation"
MODELS = ("gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-luna")
CONDITIONS = ("A", "B", "C", "D", "E", "F")
REPETITIONS = 3
SHUFFLE_SEED = 20261025
MAX_COMPLETION_TOKENS = 2048
REASONING_EFFORT = "low"
MAX_OUTPUT_WORDS = 120
MAX_NOTE_WORDS = 60
MAX_NOTES = 2

COMMON_INSTRUCTIONS = """Return exactly one JSON object with the keys \"translation\" and \"notes\". \"translation\" must be a string containing only the English poem translation, with no title and no extra poem or factual material. \"notes\" must be an array of zero to two concise strings. Keep the translation to at most 120 English words and all notes combined to at most 60 English words. Preserve the poem's core meaning, images, and emotional direction; do not invent facts. Do not add commentary outside the JSON object."""
OUTPUT_FORMAT_REMINDER = r'''Output format reminder: return a JSON object with both keys "translation" and "notes"; do not return a bare poem or prose outside JSON. Example with two verse lines: {"translation":"Line one\nLine two","notes":[]} . The backslash-n shown here must encode a line break inside the JSON string; do not put a literal line break inside a quoted JSON string.'''

CONDITION_INSTRUCTIONS = {
    "A": "Translate the following Chinese poem into English.",
    "B": "Act as a literary translator. Translate the Chinese poem into expressive, natural English with poetic language while preserving what the source says.",
    "C": "Translate for an English-only reader who does not know Chinese. Keep the translation faithful while preserving the poem's core meaning, imagery, and tone. You may use up to two brief notes for context that is needed to understand the poem.",
    "D": "Translate for English poetry reading aloud. You may adjust rhythm and restructure lines for an effective English poem, while preserving the poem's core meaning and emotional direction and adding no facts. You may use up to two brief notes for context.",
    "E": "Translate for an English-only reader who does not know Chinese. Keep the translation faithful while preserving the poem's core meaning, imagery, and tone. You may use up to two brief notes for context that is needed to understand the poem. Consider the supplied semantic notes as source interpretation, not as text to copy.",
    "F": "Translate for English poetry reading aloud. You may adjust rhythm and restructure lines for an effective English poem, while preserving the poem's core meaning and emotional direction and adding no facts. You may use up to two brief notes for context. Consider the supplied semantic notes as source interpretation, not as text to copy.",
}

RETRYABLE_API_ERRORS = {
    "api_error",
    "timeout",
    "transport_error",
    "invalid_json",
    "invalid_response",
    "model_mismatch",
    "non_text_content",
    "unexpected_finish",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w]+(?:['’][\w]+)*", text, flags=re.UNICODE))


def parse_translation(content: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the response without repairing prose or silently changing meaning."""
    parsed, errors, _, _ = _parse_translation_details(content)
    return parsed, errors


def _parse_translation_details(
    content: Any,
) -> tuple[dict[str, Any] | None, list[str], list[str], bool | None]:
    if not isinstance(content, str):
        return None, ["response content is not text"], [], None
    body = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", body, flags=re.IGNORECASE | re.DOTALL)
    format_violations: list[str] = []
    if fenced:
        body = fenced.group(1).strip()
        format_violations.append("code_fence")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        try:
            # Accept literal control characters mechanically so their exact decoded
            # text can still be scored; retain a non-conformance flag for analysis.
            parsed = json.loads(body, strict=False)
            if any(ord(char) < 0x20 for char in body):
                format_violations.append("literal_control_characters")
        except json.JSONDecodeError:
            return None, [f"invalid JSON: {exc.msg} at character {exc.pos}"], format_violations, False
    if not isinstance(parsed, dict):
        return None, ["JSON value must be an object"], format_violations, not format_violations
    errors: list[str] = []
    if set(parsed) != {"translation", "notes"}:
        errors.append("object keys must be exactly translation and notes")
    translation = parsed.get("translation")
    notes = parsed.get("notes")
    if not isinstance(translation, str) or not translation.strip():
        errors.append("translation must be a non-empty string")
    if not isinstance(notes, list) or any(not isinstance(item, str) for item in notes):
        errors.append("notes must be an array of strings")
    conformant = not format_violations
    return (parsed if not errors else None), errors, format_violations, conformant


def _constraint_violations(parsed: dict[str, Any] | None) -> list[str]:
    if parsed is None:
        return []
    violations: list[str] = []
    if len(parsed["notes"]) > MAX_NOTES:
        violations.append(f"notes_count>{MAX_NOTES}")
    if _word_count(parsed["translation"]) > MAX_OUTPUT_WORDS:
        violations.append(f"translation_words>{MAX_OUTPUT_WORDS}")
    if _word_count(" ".join(parsed["notes"])) > MAX_NOTE_WORDS:
        violations.append(f"notes_words>{MAX_NOTE_WORDS}")
    return violations


def _validate_poems(poems: Any, *, require_full_dataset: bool) -> list[dict[str, Any]]:
    if not isinstance(poems, list):
        raise ValueError("data/poems.json must contain a top-level JSON list")
    required = {
        "id", "split", "genre", "author", "title", "text", "source_url",
        "source_text_note", "semantic_notes", "critical_units", "phenomena",
    }
    seen: set[str] = set()
    for index, poem in enumerate(poems):
        if not isinstance(poem, dict):
            raise ValueError(f"poem at index {index} must be a JSON object")
        missing = required - set(poem)
        if missing:
            raise ValueError(f"poem at index {index} is missing keys: {', '.join(sorted(missing))}")
        if not isinstance(poem["id"], str) or not poem["id"].strip() or poem["id"] in seen:
            raise ValueError(f"poem at index {index} must have a unique non-empty string id")
        seen.add(poem["id"])
        if poem["split"] not in {"dev", "test"}:
            raise ValueError(f"poem {poem['id']} has unsupported split {poem['split']!r}")
        if poem["genre"] not in {"tang_shi", "song_ci"}:
            raise ValueError(f"poem {poem['id']} has unsupported genre {poem['genre']!r}")
        for field in ("author", "title", "text", "source_url", "source_text_note", "semantic_notes"):
            if not isinstance(poem[field], str):
                raise ValueError(f"poem {poem['id']} field {field} must be a string")
        if not poem["text"].strip():
            raise ValueError(f"poem {poem['id']} has empty source text")
        if not isinstance(poem["critical_units"], list):
            raise ValueError(f"poem {poem['id']} critical_units must be a list")
        unit_ids: set[str] = set()
        for unit in poem["critical_units"]:
            if not isinstance(unit, dict) or not {"id", "source_span", "meaning"} <= set(unit):
                raise ValueError(f"poem {poem['id']} has a malformed critical unit")
            if not all(isinstance(unit[key], str) and unit[key] for key in ("id", "source_span", "meaning")):
                raise ValueError(f"poem {poem['id']} critical units need non-empty string fields")
            if unit["id"] in unit_ids:
                raise ValueError(f"poem {poem['id']} has duplicate critical unit id {unit['id']!r}")
            unit_ids.add(unit["id"])
    if require_full_dataset:
        counts = {split: sum(p["split"] == split for p in poems) for split in ("dev", "test")}
        if len(poems) != 20 or counts != {"dev": 4, "test": 16}:
            raise ValueError(f"full dataset must contain 20 poems (4 dev, 16 test); found {len(poems)} ({counts})")
    return poems


def _task_id(poem_id: str, condition: str, model: str, repetition: int) -> str:
    material = "\0".join((poem_id, condition, model, str(repetition))).encode("utf-8")
    return "g_" + hashlib.sha256(material).hexdigest()[:20]


def make_manifest(
    poems: list[dict[str, Any]],
    *,
    split: str = "all",
    limit_poems: int | None = None,
    seed: int = SHUFFLE_SEED,
    repetitions: int = REPETITIONS,
    models: tuple[str, ...] | list[str] = MODELS,
) -> dict[str, Any]:
    if split not in {"all", "dev", "test"}:
        raise ValueError("split must be all, dev, or test")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError(f"models must be a non-empty unique subset of: {', '.join(MODELS)}")
    selected = [p for p in poems if split == "all" or p["split"] == split]
    if limit_poems is not None:
        if limit_poems < 1:
            raise ValueError("limit_poems must be positive")
        selected = selected[:limit_poems]
    poem_by_id = {p["id"]: p for p in selected}
    tasks: list[dict[str, Any]] = []
    for poem in selected:
        for repetition in range(repetitions):
            for model in models:
                for condition in CONDITIONS:
                    tasks.append({
                        "task_id": _task_id(poem["id"], condition, model, repetition),
                        "poem_id": poem["id"],
                        "split": poem["split"],
                        "genre": poem["genre"],
                        "model": model,
                        "condition": condition,
                        "repetition": repetition,
                    })
    random.Random(seed).shuffle(tasks)
    return {
        "manifest_version": 1,
        "shuffle_seed": seed,
        "generation": {
            "models": list(models),
            "conditions": list(CONDITIONS),
            "repetitions": repetitions,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": "omitted",
            "top_p": "omitted",
        },
        "prompt_spec": {
            "common_instructions": COMMON_INSTRUCTIONS,
            "condition_instructions": CONDITION_INSTRUCTIONS,
            "output_format_reminder": OUTPUT_FORMAT_REMINDER,
        },
        "poems": poem_by_id,
        "tasks": tasks,
    }


def prepare_manifest(
    data_path: Path = DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    split: str = "all",
    limit_poems: int | None = None,
    seed: int = SHUFFLE_SEED,
    repetitions: int = REPETITIONS,
    models: tuple[str, ...] | list[str] = MODELS,
) -> dict[str, Any]:
    with data_path.open("r", encoding="utf-8") as stream:
        poems = _validate_poems(
            json.load(stream),
            require_full_dataset=(split in {"all", "test"} and limit_poems is None),
        )
    manifest = make_manifest(
        poems,
        split=split,
        limit_poems=limit_poems,
        seed=seed,
        repetitions=repetitions,
        models=models,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_messages(
    poem: dict[str, Any],
    condition: str,
    prompt_spec: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    spec = prompt_spec or {
        "common_instructions": COMMON_INSTRUCTIONS,
        "condition_instructions": CONDITION_INSTRUCTIONS,
        "output_format_reminder": OUTPUT_FORMAT_REMINDER,
    }
    instructions = spec.get("condition_instructions", {})
    if (
        condition not in instructions
        or not isinstance(spec.get("common_instructions"), str)
        or not isinstance(spec.get("output_format_reminder"), str)
    ):
        raise ValueError(f"unknown condition {condition!r}")
    user_parts = [instructions[condition]]
    if condition in {"E", "F"}:
        user_parts.append("Semantic notes from the source-text analysis:\n" + poem["semantic_notes"])
    user_parts.append("Chinese poem:\n" + poem["text"])
    user_parts.append(spec["output_format_reminder"])
    return [
        {"role": "system", "content": spec["common_instructions"]},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _load_ledger(path: Path) -> dict[str, list[dict[str, Any]]]:
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
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                raise ValueError(f"missing task_id at {path}:{line_number}")
            attempts.setdefault(task_id, []).append(record)
    return attempts


def _retryable(record: dict[str, Any]) -> bool:
    api = record.get("api_response", {})
    if api.get("status") == "truncated":
        return True
    if api.get("status") == "error":
        return api.get("error_type") in RETRYABLE_API_ERRORS
    validation = record.get("validation", {})
    return api.get("status") == "complete" and validation.get("ok") is False


def _make_attempt_record(
    task: dict[str, Any],
    attempt_number: int,
    messages: list[dict[str, str]],
    api_response: dict[str, Any],
    *,
    max_completion_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    if api_response.get("status") == "complete":
        parsed, errors, format_violations, raw_json_conformant = _parse_translation_details(api_response.get("content"))
    else:
        parsed, errors, format_violations, raw_json_conformant = None, [], [], None
    validation = {
        "ok": parsed is not None,
        "errors": errors,
        "format_violations": format_violations,
        "raw_json_conformant": raw_json_conformant,
        "constraint_violations": _constraint_violations(parsed),
        "translation": parsed.get("translation") if parsed is not None else None,
        "notes": parsed.get("notes") if parsed is not None else None,
        "translation_word_count": _word_count(parsed["translation"]) if parsed is not None else None,
        "notes_word_count": _word_count(" ".join(parsed["notes"])) if parsed is not None else None,
    }
    return {
        **task,
        "attempt": attempt_number,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request": {
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
            "temperature": "omitted",
            "top_p": "omitted",
        },
        "api_response": api_response,
        "validation": validation,
    }


def _append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()


def run_generation(
    manifest_path: Path = DEFAULT_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    *,
    split: str | None = None,
    workers: int = 3,
    max_retries: int = 2,
    request_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, int]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be between 0 and 2")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("manifest_version") != 1:
        raise ValueError("unsupported generation manifest version")
    poems = manifest.get("poems")
    tasks = manifest.get("tasks")
    if not isinstance(poems, dict) or not isinstance(tasks, list):
        raise ValueError("manifest must contain poem and task collections")
    selected_tasks = [task for task in tasks if split is None or task.get("split") == split]
    counts = {"selected": len(selected_tasks), "completed_new": 0, "completed_existing": 0, "failed": 0}
    if not selected_tasks:
        return counts
    generation = manifest.get("generation", {})
    prompt_spec = manifest.get("prompt_spec")
    if not isinstance(prompt_spec, dict) or not isinstance(prompt_spec.get("common_instructions"), str):
        raise ValueError("manifest must freeze common and condition-specific prompts")
    max_completion_tokens = generation.get("max_completion_tokens")
    reasoning_effort = generation.get("reasoning_effort")
    if type(max_completion_tokens) is not int or max_completion_tokens < 1 or not isinstance(reasoning_effort, str):
        raise ValueError("manifest has invalid frozen generation parameters")
    if generation.get("temperature") != "omitted" or generation.get("top_p") != "omitted":
        raise ValueError("manifest must record temperature and top_p as omitted")
    by_split: dict[str, list[dict[str, Any]]] = {}
    for task in selected_tasks:
        if task.get("split") not in {"dev", "test"} or task.get("poem_id") not in poems:
            raise ValueError(f"invalid task in manifest: {task!r}")
        by_split.setdefault(task["split"], []).append(task)
    results_dir.mkdir(parents=True, exist_ok=True)
    ledgers = {name: _load_ledger(results_dir / f"{name}.jsonl") for name in by_split}
    locks = {name: threading.Lock() for name in by_split}
    if request_fn is None:
        api_client.validate_environment()
        request_fn = api_client.request
    count_lock = threading.Lock()
    expected_requests: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        messages = build_messages(poems[task["poem_id"]], task["condition"], prompt_spec)
        expected = {
            "task": task,
            "request": {
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                "reasoning_effort": reasoning_effort,
                "temperature": "omitted",
                "top_p": "omitted",
            },
        }
        expected_requests[task["task_id"]] = expected
        for old_record in ledgers[task["split"]].get(task["task_id"], []):
            if any(old_record.get(key) != value for key, value in task.items()):
                raise ValueError(f"stored generation task metadata differs for {task['task_id']}")
            if old_record.get("request") != expected["request"]:
                raise ValueError(f"stored prompt or parameters differ for {task['task_id']}; refusing to mix runs")

    def process(task: dict[str, Any]) -> str:
        split_name = task["split"]
        previous = ledgers[split_name].get(task["task_id"], [])
        if any(item.get("validation", {}).get("ok") is True for item in previous):
            return "completed_existing"
        if previous and (len(previous) > max_retries or not _retryable(previous[-1])):
            return "failed"
        messages = expected_requests[task["task_id"]]["request"]["messages"]
        for attempt_number in range(len(previous) + 1, max_retries + 2):
            response = request_fn(
                task["model"],
                messages,
                max_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
            record = _make_attempt_record(
                task,
                attempt_number,
                messages,
                response,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
            _append_jsonl(results_dir / f"{split_name}.jsonl", record, locks[split_name])
            previous.append(record)
            if record["validation"]["ok"]:
                return "completed_new"
            if attempt_number > max_retries or not _retryable(record):
                return "failed"
        return "failed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, task) for task in selected_tasks]
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            with count_lock:
                counts[outcome] += 1
    return counts
