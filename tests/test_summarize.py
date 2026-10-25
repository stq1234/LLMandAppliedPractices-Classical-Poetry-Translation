import json
import tempfile
import unittest
from pathlib import Path

from src.generate import MODELS, make_manifest
from src.summarize import summarize_results


def poems():
    result = []
    for index in range(2):
        result.append({
            "id": f"p{index}",
            "split": "dev",
            "genre": "tang_shi" if index == 0 else "song_ci",
            "author": "Example",
            "title": f"Poem {index}",
            "text": "春晓。",
            "source_url": "https://example.invalid/poem",
            "source_text_note": "Fixture",
            "semantic_notes": "Spring dawn.",
            "critical_units": [{"id": "u1", "source_span": "春晓", "meaning": "spring dawn"}],
            "phenomena": [],
        })
    return result


class SummarizeTests(unittest.TestCase):
    def test_poem_level_pairing_and_generation_diagnostics(self):
        manifest = make_manifest(poems(), split="dev", repetitions=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            generation_dir = root / "generation"
            generation_dir.mkdir()
            generation_records = []
            score_records = []
            for task in manifest["tasks"]:
                generation_records.append({
                    **task,
                    "attempt": 1,
                    "validation": {
                        "ok": True,
                        "raw_json_conformant": True,
                        "format_violations": [],
                        "constraint_violations": [],
                        "translation_word_count": 4,
                        "notes_word_count": 0,
                        "notes": [],
                    },
                })
            (generation_dir / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in generation_records), encoding="utf-8")

            task_map = {task["task_id"]: task for task in manifest["tasks"]}
            for poem in poems():
                selected = [task for task in manifest["tasks"] if task["poem_id"] == poem["id"] and task["condition"] in {"A", "B"}]
                mapping = []
                evaluations = []
                for index, task in enumerate(selected):
                    opaque_id = f"opaque-{poem['id']}-{index}"
                    mapping.append({
                        "opaque_id": opaque_id,
                        "task_id": task["task_id"],
                        "model": task["model"],
                        "condition": task["condition"],
                    })
                    score = 4 if task["condition"] == "A" else 5
                    evaluations.append({
                        "opaque_id": opaque_id,
                        "scores": {name: score for name in ("meaning", "imagery", "tone", "english", "faithful_fit", "poetic_fit")},
                        "critical_units": [],
                    })
                score_records.append({
                    "variant": "primary",
                    "judge_model": "gpt-6.1-sol",
                    "batch_id": f"{poem['id']}__r0__b00",
                    "request": {"opaque_mapping": mapping},
                    "validation": {"ok": True, "evaluations": evaluations},
                })
            judging_dir = root / "judging"
            judging_dir.mkdir()
            (judging_dir / "primary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in score_records), encoding="utf-8")
            output = root / "summary"
            result = summarize_results(
                manifest_path,
                generation_dir,
                judging_dir,
                output,
                bootstrap_samples=50,
            )
            self.assertEqual(result["expected_generation_tasks"], 36)
            self.assertEqual(result["successful_generation_tasks"], 36)
            self.assertEqual(result["judge_score_rows"], 12)
            with (output / "poem_paired_changes.csv").open(encoding="utf-8-sig", newline="") as stream:
                paired = list(__import__("csv").DictReader(stream))
            self.assertEqual(len(paired), 12)
            self.assertTrue(all(float(row["delta_b_minus_a"]) == 1.0 for row in paired))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["bootstrap"]["unit"], "poem")
            self.assertTrue((output / "summary_long.csv").exists())
            self.assertTrue((output / "generation_long.csv").exists())


if __name__ == "__main__":
    unittest.main()
