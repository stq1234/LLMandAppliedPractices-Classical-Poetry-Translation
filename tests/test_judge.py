import json
import tempfile
import unittest
from pathlib import Path

from src.generate import CONDITIONS, MODELS, SHUFFLE_SEED, _task_id, make_manifest
from src.judge import (
    DIMENSIONS,
    build_judge_messages,
    build_judge_plan,
    parse_judgment,
    run_judging,
)


def sample_poems():
    poems = []
    for index in range(20):
        split = "dev" if index < 4 else "test"
        poems.append({
            "id": f"p{index:02d}",
            "split": split,
            "genre": "tang_shi" if index % 2 == 0 else "song_ci",
            "author": "Example",
            "title": f"Poem {index}",
            "text": "春晓，花开。",
            "source_url": "https://example.invalid/poem",
            "source_text_note": "Fixture",
            "semantic_notes": "Spring dawn; flowers bloom.",
            "critical_units": [{"id": "u1", "source_span": "春晓", "meaning": "spring dawn"}],
            "phenomena": ["imagery"],
        })
    return poems


def write_complete_generation(poems, path):
    records = []
    for poem in poems:
        if poem["split"] != "test":
            continue
        for repetition in range(3):
            for model in MODELS:
                for condition in CONDITIONS:
                    task_id = _task_id(poem["id"], condition, model, repetition)
                    records.append({
                        "task_id": task_id,
                        "poem_id": poem["id"],
                        "split": "test",
                        "genre": poem["genre"],
                        "model": model,
                        "condition": condition,
                        "repetition": repetition,
                        "validation": {"ok": True, "translation": "Spring dawn, flowers bloom.", "notes": []},
                    })
    path.mkdir(parents=True, exist_ok=True)
    (path / "test.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def make_response(messages):
    payload = json.loads(messages[1]["content"])
    evaluations = []
    unit_id = payload["critical_units"][0]["id"]
    for item in payload["translations_to_score"]:
        evaluations.append({
            "opaque_id": item["opaque_id"],
            "scores": {dimension: 4 for dimension in DIMENSIONS},
            "critical_units": [{
                "id": unit_id,
                "status": "present",
                "quote": "Spring dawn",
                "explanation": "The translation conveys the morning setting.",
            }],
        })
    return {"status": "complete", "content": json.dumps({"evaluations": evaluations}), "finish_reason": "stop"}


class JudgeTests(unittest.TestCase):
    def test_variant_selection_and_reverse_order_preserve_batch_mapping(self):
        poems = sample_poems()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "poems.json"
            data_path.write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
            generation_dir = root / "generation"
            write_complete_generation(poems, generation_dir)
            primary = build_judge_plan(variant="primary", generation_results_dir=generation_dir, data_path=data_path)
            secondary = build_judge_plan(variant="secondary", generation_results_dir=generation_dir, data_path=data_path)
            reversed_plan = build_judge_plan(variant="reversed", generation_results_dir=generation_dir, data_path=data_path)
            self.assertEqual(len(primary["batches"]), 144)
            self.assertEqual(len(secondary["batches"]), 48)
            self.assertEqual(len(reversed_plan["batches"]), 12)
            primary_rep0 = {batch["batch_id"]: [item["opaque_id"] for item in batch["items"]]
                            for batch in primary["batches"] if batch["repetition"] == 0}
            secondary_order = {batch["batch_id"]: [item["opaque_id"] for item in batch["items"]]
                               for batch in secondary["batches"]}
            reverse_order = {batch["batch_id"]: [item["opaque_id"] for item in batch["items"]]
                             for batch in reversed_plan["batches"]}
            self.assertEqual(secondary_order, {key: value for key, value in primary_rep0.items()})
            for batch_id, values in reverse_order.items():
                self.assertEqual(values, list(reversed(primary_rep0[batch_id])))
            self.assertEqual(secondary["judge_model"], "gpt-5.6-luna")
            self.assertEqual(reversed_plan["judge_model"], "gpt-6.1-sol")

    def test_judge_prompt_is_blind_to_generation_metadata(self):
        poems = sample_poems()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "poems.json"
            data_path.write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
            generation_dir = root / "generation"
            write_complete_generation(poems, generation_dir)
            plan = build_judge_plan(variant="secondary", generation_results_dir=generation_dir, data_path=data_path)
            batch = plan["batches"][0]
            messages = build_judge_messages(plan["source_poems"][batch["poem_id"]], {
                **batch,
                "critical_units": plan["source_poems"][batch["poem_id"]]["critical_units"],
            }, plan["judge_instructions"])
            self.assertNotIn("gpt-6.1-sol", messages[1]["content"])
            self.assertNotIn("condition", messages[1]["content"])
            self.assertNotIn("task_id", messages[1]["content"])
            self.assertEqual(set(json.loads(messages[1]["content"])), {"source_poem_zh", "critical_units", "translations_to_score"})

    def test_generation_manifest_scopes_dev_to_twelve_primary_batches(self):
        poems = sample_poems()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "poems.json"
            data_path.write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
            generation_manifest = make_manifest(poems, split="dev", repetitions=1)
            manifest_path = root / "dev_manifest.json"
            manifest_path.write_text(json.dumps(generation_manifest), encoding="utf-8")
            plan = build_judge_plan(
                variant="primary",
                split="dev",
                generation_results_dir=root / "empty_generation",
                generation_manifest_path=manifest_path,
                data_path=data_path,
            )
            self.assertEqual(len(plan["batches"]), 12)
            self.assertTrue(all(batch["split"] == "dev" for batch in plan["batches"]))

    def test_response_validation_checks_opaque_ids_units_and_verbatim_quotes(self):
        poems = sample_poems()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "poems.json"
            data_path.write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
            generation_dir = root / "generation"
            write_complete_generation(poems, generation_dir)
            plan = build_judge_plan(variant="secondary", generation_results_dir=generation_dir, data_path=data_path)
            batch = dict(plan["batches"][0])
            batch["critical_units"] = plan["source_poems"][batch["poem_id"]]["critical_units"]
            valid = make_response(build_judge_messages(plan["source_poems"][batch["poem_id"]], batch))
            parsed, errors = parse_judgment(valid["content"], batch)
            self.assertIsNotNone(parsed)
            self.assertEqual(errors, [])
            bad = json.loads(valid["content"])
            bad["evaluations"][0]["critical_units"][0]["quote"] = "not copied"
            parsed, errors = parse_judgment(json.dumps(bad), batch)
            self.assertIsNone(parsed)
            self.assertTrue(any("exact translation/note substring" in error for error in errors))
            missing_ok = json.loads(valid["content"])
            missing_ok["evaluations"][0]["critical_units"][0].update({
                "status": "missing", "quote": "", "explanation": "No part of this meaning appears in the translation.",
            })
            parsed, errors = parse_judgment(json.dumps(missing_ok), batch)
            self.assertIsNotNone(parsed)
            self.assertEqual(errors, [])

    def test_judging_saves_raw_response_and_resumes_without_replacement(self):
        poems = sample_poems()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "poems.json"
            data_path.write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
            generation_dir = root / "generation"
            write_complete_generation(poems, generation_dir)
            plan = build_judge_plan(variant="secondary", generation_results_dir=generation_dir, data_path=data_path)
            plan["batches"] = plan["batches"][:1]
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            calls = []

            def fake_request(model, messages, **kwargs):
                calls.append((model, kwargs))
                return make_response(messages)

            results_dir = root / "judging"
            outcome = run_judging(plan_path, results_dir, request_fn=fake_request)
            self.assertEqual(outcome["complete"], 1)
            self.assertEqual(calls[0][0], "gpt-5.6-luna")
            self.assertEqual(calls[0][1], {"max_tokens": 8192, "reasoning_effort": "low"})
            record = json.loads((results_dir / "secondary.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(record["validation"]["ok"])
            self.assertEqual(record["api_response"]["status"], "complete")
            outcome = run_judging(plan_path, results_dir, request_fn=fake_request)
            self.assertEqual(outcome["skipped"], 1)
            self.assertEqual(len(calls), 1)

    def test_failed_and_exhausted_batches_remain_failures_and_model_mismatch_is_not_retried(self):
        poems = sample_poems()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "poems.json"
            data_path.write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
            generation_dir = root / "generation"
            write_complete_generation(poems, generation_dir)
            plan = build_judge_plan(variant="secondary", generation_results_dir=generation_dir, data_path=data_path)
            plan["batches"] = plan["batches"][:1]
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            mismatch_calls = []

            def mismatch_request(model, messages, **kwargs):
                mismatch_calls.append(model)
                return {"status": "error", "error_type": "model_mismatch", "error": "wrong model"}

            mismatch_dir = root / "mismatch"
            first = run_judging(plan_path, mismatch_dir, request_fn=mismatch_request)
            self.assertEqual((first["failed"], first["skipped"]), (1, 0))
            stored = json.loads((mismatch_dir / "secondary.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(stored["api_response"]["error_type"], "model_mismatch")
            resumed = run_judging(plan_path, mismatch_dir, request_fn=mismatch_request)
            self.assertEqual((resumed["failed"], resumed["skipped"]), (1, 0))
            self.assertEqual(mismatch_calls, ["gpt-5.6-luna"])

            retry_calls = []

            def retryable_request(model, messages, **kwargs):
                retry_calls.append(model)
                return {"status": "error", "error_type": "invalid_json", "error": "bad response"}

            exhausted_dir = root / "exhausted"
            first = run_judging(plan_path, exhausted_dir, max_retries=0, request_fn=retryable_request)
            self.assertEqual((first["failed"], first["skipped"]), (1, 0))
            resumed = run_judging(plan_path, exhausted_dir, max_retries=0, request_fn=retryable_request)
            self.assertEqual((resumed["failed"], resumed["skipped"]), (1, 0))
            self.assertEqual(retry_calls, ["gpt-5.6-luna"])


if __name__ == "__main__":
    unittest.main()
