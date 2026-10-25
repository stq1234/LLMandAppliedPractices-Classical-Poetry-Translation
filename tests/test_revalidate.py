import json
import tempfile
import unittest
from pathlib import Path

from src.judge import DIMENSIONS, parse_judgment
from src.revalidate import QUOTE_MATCH_POLICY, revalidate_results, revalidate_variant_records


def fixture_batch():
    return {
        "items": [{
            "opaque_id": "opaque-a",
            "translation": "Spring\n dawn, flowers bloom.",
            "notes": [],
        }],
        "critical_units": [{"id": "u1", "source_span": "春晓", "meaning": "spring dawn"}],
    }


def make_content(quote, score=4):
    return json.dumps({
        "evaluations": [{
            "opaque_id": "opaque-a",
            "scores": {dimension: score for dimension in DIMENSIONS},
            "critical_units": [{
                "id": "u1",
                "status": "present",
                "quote": quote,
                "explanation": "The source meaning is represented.",
            }],
        }],
    })


def make_record(attempt, quote, score=4):
    batch = fixture_batch()
    user_payload = {
        "source_poem_zh": "春晓，花开。",
        "critical_units": batch["critical_units"],
        "translations_to_score": batch["items"],
    }
    content = make_content(quote, score)
    parsed, errors = parse_judgment(content, batch)
    return {
        "judge_task_id": "primary:gpt-6.1-sol:p00__r0__b00",
        "variant": "primary",
        "judge_model": "gpt-6.1-sol",
        "batch_id": "p00__r0__b00",
        "attempt": attempt,
        "request": {"messages": [
            {"role": "system", "content": "Frozen evaluation instructions."},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]},
        "api_response": {"status": "complete", "content": content},
        "validation": {
            "ok": parsed is not None,
            "errors": errors,
            "evaluations": parsed["evaluations"] if parsed else None,
        },
    }


class RevalidationTests(unittest.TestCase):
    def test_normalized_quote_matching_folds_whitespace_and_case_only(self):
        batch = fixture_batch()
        content = make_content(" spring\tDAWN ")
        strict, strict_errors = parse_judgment(content, batch)
        normalized, normalized_errors = parse_judgment(content, batch, normalize_quotes=True)
        self.assertIsNone(strict)
        self.assertTrue(any("not an exact translation/note substring" in error for error in strict_errors))
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized_errors, [])

    def test_normalized_quote_matching_does_not_rewrite_punctuation_or_accept_fake_text(self):
        batch = fixture_batch()
        for quote in ("Spring dawn; flowers", "Moonlit dawn"):
            with self.subTest(quote=quote):
                parsed, errors = parse_judgment(make_content(quote), batch, normalize_quotes=True)
                self.assertIsNone(parsed)
                self.assertTrue(any("not an exact translation/note substring" in error for error in errors))

    def test_literal_backslash_n_is_not_treated_as_whitespace(self):
        batch = fixture_batch()
        batch["items"][0]["translation"] = r"Spring\n dawn, flowers bloom."
        parsed, errors = parse_judgment(make_content("Spring dawn"), batch, normalize_quotes=True)
        self.assertIsNone(parsed)
        self.assertTrue(any("not an exact translation/note substring" in error for error in errors))

    def test_blank_nonmissing_quote_is_rejected_after_normalization(self):
        batch = fixture_batch()
        content = json.loads(make_content(" \t\n "))
        parsed, errors = parse_judgment(json.dumps(content), batch, normalize_quotes=True)
        self.assertIsNone(parsed)
        self.assertTrue(any("needs evidence for status present" in error for error in errors))

    def test_empty_quote_remains_valid_for_missing_unit(self):
        batch = fixture_batch()
        content = json.loads(make_content(""))
        content["evaluations"][0]["critical_units"][0]["status"] = "missing"
        parsed, errors = parse_judgment(json.dumps(content), batch, normalize_quotes=True)
        self.assertIsNotNone(parsed)
        self.assertEqual(errors, [])

    def test_revalidation_selects_earliest_normalized_attempt_and_preserves_source(self):
        records = [
            make_record(1, "spring dawn", score=4),
            make_record(2, "Spring\n dawn", score=5),
        ]
        original_contents = [record["api_response"]["content"] for record in records]
        accepted, summary = revalidate_variant_records(records)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source_attempt"], 1)
        self.assertEqual(accepted[0]["original_validation"]["ok"], False)
        self.assertEqual(accepted[0]["validation"]["evaluations"][0]["scores"]["meaning"], 4)
        self.assertEqual(accepted[0]["quote_match_policy"], QUOTE_MATCH_POLICY)
        self.assertEqual(summary["strict_valid_batches"], 1)
        self.assertEqual(summary["accepted_batches"], 1)
        self.assertEqual(summary["selected_attempts_rejected_by_strict_policy"], 1)
        self.assertEqual(summary["selected_attempt_changed_batches"], 1)
        self.assertEqual(summary["selected_score_payload_changed_batches"], 1)
        self.assertEqual([record["api_response"]["content"] for record in records], original_contents)

    def test_directory_output_has_one_accepted_record_and_leaves_raw_file_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "raw"
            output_dir = root / "accepted"
            input_dir.mkdir()
            records = [make_record(1, "spring dawn"), make_record(2, "Spring\n dawn")]
            raw_content = "".join(json.dumps(record) + "\n" for record in records)
            raw_path = input_dir / "primary.jsonl"
            raw_path.write_text(raw_content, encoding="utf-8")

            summary = revalidate_results(
                input_dir,
                output_dir,
                variants=("primary",),
                expected_batch_counts={"primary": 1},
            )

            accepted = [json.loads(line) for line in (output_dir / "primary.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["source_attempt"], 1)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), raw_content)
            self.assertEqual(summary["variants"]["primary"]["accepted_batches"], 1)
            self.assertTrue((output_dir / "revalidation_summary.json").is_file())

    def test_directory_revalidation_rejects_batch_with_only_fabricated_quotes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "raw"
            input_dir.mkdir()
            raw_path = input_dir / "primary.jsonl"
            raw_path.write_text(json.dumps(make_record(1, "Moonlit dawn")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without a valid attempt"):
                revalidate_results(
                    input_dir,
                    root / "accepted",
                    variants=("primary",),
                    expected_batch_counts={"primary": 1},
                )
            self.assertFalse((root / "accepted").exists())


if __name__ == "__main__":
    unittest.main()
