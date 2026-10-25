"""Build accepted judge files from unchanged raw attempts using one quote policy."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .judge import QUOTE_MATCH_POLICY, parse_judgment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "results" / "judging"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "judging_accepted"
VARIANTS = ("primary", "secondary", "reversed")
EXPECTED_BATCH_COUNTS = {"primary": 144, "secondary": 48, "reversed": 12}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"expected an object at {path}:{line_number}")
            records.append(record)
    return records


def _batch_from_request(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[1], dict):
        raise ValueError(f"judge attempt {record.get('judge_task_id')} has no frozen user request")
    user_content = messages[1].get("content")
    try:
        payload = json.loads(user_content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"judge attempt {record.get('judge_task_id')} has an invalid frozen user payload") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"judge attempt {record.get('judge_task_id')} has a non-object user payload")
    items = payload.get("translations_to_score")
    units = payload.get("critical_units")
    if not isinstance(items, list) or not isinstance(units, list):
        raise ValueError(f"judge attempt {record.get('judge_task_id')} lacks translation or critical-unit data")
    return {"items": items, "critical_units": units}


def _parsed_attempt(
    record: dict[str, Any],
    batch: dict[str, Any],
    *,
    normalize_quotes: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    response = record.get("api_response")
    if not isinstance(response, dict) or response.get("status") != "complete":
        return None, []
    return parse_judgment(response.get("content"), batch, normalize_quotes=normalize_quotes)


def _score_signature(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if parsed is None:
        return None
    return {
        evaluation["opaque_id"]: evaluation["scores"]
        for evaluation in parsed["evaluations"]
    }


def revalidate_variant_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the earliest attempt passing the shared normalized-quote validation."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for file_order, record in enumerate(records):
        task_id = record.get("judge_task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("judge attempt is missing judge_task_id")
        grouped[task_id].append((file_order, record))

    accepted: list[dict[str, Any]] = []
    strict_valid_attempts = 0
    normalized_valid_attempts = 0
    strict_valid_batches = 0
    recovered_batches = 0
    changed_selection_batches = 0
    relaxed_selected_batches = 0
    changed_score_batches = 0
    stored_validation_mismatches = 0
    unresolved_batches: list[str] = []

    for task_id in sorted(grouped):
        attempts = sorted(
            grouped[task_id],
            key=lambda pair: (
                pair[1].get("attempt") if type(pair[1].get("attempt")) is int else pair[0] + 1,
                pair[0],
            ),
        )
        expected_request = attempts[0][1].get("request")
        for _, record in attempts[1:]:
            if record.get("request") != expected_request:
                raise ValueError(f"stored request differs across attempts for {task_id}")

        strict_successes: list[tuple[dict[str, Any], dict[str, Any]]] = []
        normalized_successes: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for _, record in attempts:
            batch = _batch_from_request(record)
            strict_parsed, _ = _parsed_attempt(record, batch, normalize_quotes=False)
            normalized_parsed, _ = _parsed_attempt(record, batch, normalize_quotes=True)
            strict_ok = strict_parsed is not None
            normalized_ok = normalized_parsed is not None
            strict_valid_attempts += int(strict_ok)
            normalized_valid_attempts += int(normalized_ok)
            stored_ok = record.get("validation", {}).get("ok") is True
            stored_validation_mismatches += int(stored_ok != strict_ok)
            if strict_parsed is not None:
                strict_successes.append((record, strict_parsed))
            if normalized_parsed is not None:
                normalized_successes.append((record, normalized_parsed))

        if strict_successes:
            strict_valid_batches += 1
        if not normalized_successes:
            unresolved_batches.append(task_id)
            continue

        selected_record, selected_parsed = normalized_successes[0]
        strict_selection = strict_successes[0] if strict_successes else None
        recovered_batches += int(strict_selection is None)
        selected_was_strictly_invalid = not any(
            selected_record is record for record, _ in strict_successes
        )
        relaxed_selected_batches += int(selected_was_strictly_invalid)
        if strict_selection is not None:
            strict_record, strict_parsed = strict_selection
            changed_selection_batches += int(strict_record.get("attempt") != selected_record.get("attempt"))
            changed_score_batches += int(
                _score_signature(strict_parsed) != _score_signature(selected_parsed)
            )

        derived = dict(selected_record)
        derived["source_attempt"] = selected_record.get("attempt")
        derived["original_validation"] = copy.deepcopy(selected_record.get("validation"))
        derived["quote_match_policy"] = QUOTE_MATCH_POLICY
        derived["validation"] = {
            "ok": True,
            "errors": [],
            "evaluations": selected_parsed["evaluations"],
            "quote_match_policy": QUOTE_MATCH_POLICY,
        }
        accepted.append(derived)

    summary = {
        "attempts": len(records),
        "unique_batches": len(grouped),
        "strict_valid_attempts": strict_valid_attempts,
        "normalized_valid_attempts": normalized_valid_attempts,
        "strict_valid_batches": strict_valid_batches,
        "accepted_batches": len(accepted),
        "recovered_batches": recovered_batches,
        "selected_attempts_rejected_by_strict_policy": relaxed_selected_batches,
        "selected_attempt_changed_batches": changed_selection_batches,
        "selected_score_payload_changed_batches": changed_score_batches,
        "stored_strict_validation_mismatches": stored_validation_mismatches,
        "unresolved_batches": unresolved_batches,
        "quote_match_policy": QUOTE_MATCH_POLICY,
    }
    return accepted, summary


def revalidate_results(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    variants: tuple[str, ...] = VARIANTS,
    expected_batch_counts: dict[str, int] | None = EXPECTED_BATCH_COUNTS,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("accepted files must be written to a separate output directory")
    summaries: dict[str, Any] = {}
    outputs: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unsupported judge variant: {variant}")
        source = input_dir / f"{variant}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"missing raw judge attempts: {source}")
        records = _read_jsonl(source)
        accepted, summary = revalidate_variant_records(records)
        expected = expected_batch_counts.get(variant) if expected_batch_counts is not None else None
        if expected is not None and summary["unique_batches"] != expected:
            raise ValueError(
                f"{variant} has {summary['unique_batches']} batches; expected {expected} before acceptance"
            )
        if summary["unresolved_batches"]:
            raise ValueError(
                f"{variant} has {len(summary['unresolved_batches'])} batches without a valid attempt"
            )
        summaries[variant] = summary
        outputs[variant] = accepted

    output_dir.mkdir(parents=True, exist_ok=True)
    for variant, accepted in outputs.items():
        content = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in accepted)
        (output_dir / f"{variant}.jsonl").write_text(content, encoding="utf-8", newline="\n")
    summary = {"quote_match_policy": QUOTE_MATCH_POLICY, "variants": summaries}
    (output_dir / "revalidation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    summary = revalidate_results(args.input_dir, args.output_dir, variants=tuple(args.variants))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
