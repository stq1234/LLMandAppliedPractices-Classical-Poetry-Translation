import json
import tempfile
import unittest
from pathlib import Path

from src.generate import (
    CONDITIONS,
    MODELS,
    build_messages,
    make_manifest,
    parse_translation,
    run_generation,
    _parse_translation_details,
)


def sample_poem(poem_id="p1", split="dev"):
    return {
        "id": poem_id,
        "split": split,
        "genre": "tang_shi",
        "author": "Example",
        "title": "Example poem",
        "text": "春眠不觉晓，处处闻啼鸟。",
        "source_url": "https://example.invalid/poem",
        "source_text_note": "Test fixture",
        "semantic_notes": "Spring dawn and birdsong.",
        "critical_units": [{"id": "u1", "source_span": "春眠", "meaning": "sleep through spring dawn"}],
        "phenomena": ["imagery"],
    }


class GenerationTests(unittest.TestCase):
    def test_parser_accepts_a_json_fence_and_rejects_extra_prose(self):
        parsed, errors = parse_translation('```json\n{"translation":"At dawn, I sleep through spring.","notes":[]}\n```')
        self.assertEqual(errors, [])
        self.assertEqual(parsed["translation"], "At dawn, I sleep through spring.")
        parsed, errors = parse_translation('Here is the translation: {"translation":"x","notes":[]}')
        self.assertIsNone(parsed)
        self.assertTrue(errors)

    def test_parser_enforces_schema_and_word_limits(self):
        parsed, errors = parse_translation(json.dumps({"translation": "dawn " * 121, "notes": []}))
        self.assertIsNotNone(parsed)
        self.assertEqual(errors, [])
        parsed, errors = parse_translation('{"translation":"ok","notes":[],"extra":"value"}')
        self.assertIsNone(parsed)
        self.assertIn("object keys must be exactly translation and notes", errors)

    def test_manifest_has_stable_complete_factorial_and_seeded_shuffle(self):
        poems = [sample_poem()]
        first = make_manifest(poems)
        second = make_manifest(poems)
        self.assertEqual(first, second)
        self.assertEqual(len(first["tasks"]), 3 * len(MODELS) * len(CONDITIONS))
        task_keys = {(t["poem_id"], t["model"], t["condition"], t["repetition"]) for t in first["tasks"]}
        self.assertEqual(len(task_keys), len(first["tasks"]))
        self.assertEqual(first["generation"]["temperature"], "omitted")
        self.assertEqual(first["generation"]["top_p"], "omitted")
        self.assertIn("common_instructions", first["prompt_spec"])
        self.assertEqual(first["prompt_spec"]["condition_instructions"]["A"], "Translate the following Chinese poem into English.")

    def test_prompts_only_supply_semantics_in_e_and_f(self):
        poem = sample_poem()
        for condition in "ABCD":
            self.assertNotIn(poem["semantic_notes"], build_messages(poem, condition)[1]["content"])
        for condition in "EF":
            self.assertIn(poem["semantic_notes"], build_messages(poem, condition)[1]["content"])
        self.assertTrue(build_messages(poem, "A")[1]["content"].endswith("do not put a literal line break inside a quoted JSON string."))
        self.assertIn(r'Line one\nLine two', build_messages(poem, "A")[1]["content"])

    def test_literal_newlines_and_fences_are_accepted_but_flagged(self):
        raw_newline = '{"translation":"Line one\nLine two","notes":[]}'
        parsed, errors, violations, conformant = _parse_translation_details(raw_newline)
        self.assertEqual(errors, [])
        self.assertEqual(parsed["translation"], "Line one\nLine two")
        self.assertEqual(violations, ["literal_control_characters"])
        self.assertFalse(conformant)
        fenced = '```json\n{"translation":"Spring dawn.","notes":[]}\n```'
        parsed, errors, violations, conformant = _parse_translation_details(fenced)
        self.assertEqual(errors, [])
        self.assertEqual(violations, ["code_fence"])
        self.assertFalse(conformant)

    def test_invalid_format_retries_twice_then_success_is_not_replaced(self):
        poem = sample_poem()
        manifest = make_manifest([poem], repetitions=1)
        # Exercise one fixed task while preserving a real manifest entry.
        task = manifest["tasks"][0]
        manifest["tasks"] = [task]
        calls = []

        def fake_request(model, messages, **kwargs):
            calls.append((model, messages, kwargs))
            if len(calls) < 3:
                return {"status": "complete", "content": "not JSON", "finish_reason": "stop"}
            return {"status": "complete", "content": '{"translation":"At dawn, spring sleep ends.","notes":[]}', "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            results = root / "results"
            outcome = run_generation(manifest_path, results, request_fn=fake_request, workers=2)
            self.assertEqual(outcome, {"selected": 1, "completed_new": 1, "completed_existing": 0, "failed": 0})
            records = [json.loads(line) for line in (results / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["attempt"] for r in records], [1, 2, 3])
            self.assertTrue(records[-1]["validation"]["ok"])
            self.assertEqual(records[-1]["request"]["reasoning_effort"], "low")
            outcome = run_generation(manifest_path, results, request_fn=fake_request, workers=2)
            self.assertEqual(outcome["completed_existing"], 1)
            self.assertEqual(len(calls), 3)

    def test_constraint_violations_are_recorded_without_resampling(self):
        poem = sample_poem()
        manifest = make_manifest([poem], repetitions=1)
        task = manifest["tasks"][0]
        manifest["tasks"] = [task]
        calls = []

        def fake_request(model, messages, **kwargs):
            calls.append(1)
            payload = {"translation": "spring " * 121, "notes": ["note"] * 3}
            return {"status": "complete", "content": json.dumps(payload), "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            results = root / "results"
            outcome = run_generation(manifest_path, results, request_fn=fake_request)
            self.assertEqual(outcome["completed_new"], 1)
            self.assertEqual(len(calls), 1)
            record = json.loads((results / "dev.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(record["validation"]["ok"])
            self.assertEqual(record["validation"]["constraint_violations"], ["notes_count>2", "translation_words>120"])

    def test_resume_refuses_a_changed_frozen_prompt(self):
        poem = sample_poem()
        manifest = make_manifest([poem], repetitions=1)
        task = manifest["tasks"][0]
        manifest["tasks"] = [task]

        def fake_request(model, messages, **kwargs):
            return {"status": "complete", "content": '{"translation":"Spring dawn.","notes":[]}', "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            results = root / "results"
            run_generation(manifest_path, results, request_fn=fake_request)
            changed = dict(manifest)
            changed["prompt_spec"] = dict(manifest["prompt_spec"])
            changed["prompt_spec"]["common_instructions"] += " changed"
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stored prompt or parameters differ"):
                run_generation(manifest_path, results, request_fn=fake_request)

    def test_splits_write_to_separate_ledgers(self):
        dev = sample_poem("dev-poem", "dev")
        test = sample_poem("test-poem", "test")
        manifest = make_manifest([dev, test], repetitions=1)
        manifest["tasks"] = [t for t in manifest["tasks"] if t["condition"] == "A" and t["model"] == MODELS[0]]

        def fake_request(model, messages, **kwargs):
            return {"status": "complete", "content": '{"translation":"Spring comes at dawn.","notes":[]}', "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            results = root / "results"
            outcome = run_generation(manifest_path, results, request_fn=fake_request)
            self.assertEqual(outcome["completed_new"], 2)
            self.assertTrue((results / "dev.jsonl").exists())
            self.assertTrue((results / "test.jsonl").exists())
            dev_record = json.loads((results / "dev.jsonl").read_text(encoding="utf-8"))
            test_record = json.loads((results / "test.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(dev_record["split"], "dev")
            self.assertEqual(test_record["split"], "test")


if __name__ == "__main__":
    unittest.main()
