"""Command-line entry point for preparing and running the experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import (
    DATA_PATH,
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS_DIR,
    MODELS,
    SHUFFLE_SEED,
    prepare_manifest,
    run_generation,
)
from .judge import (
    BATCH_SIZE,
    DEFAULT_JUDGE_RESULTS_DIR,
    build_judge_plan,
    run_judging,
)
from .summarize import DEFAULT_SUMMARY_DIR, summarize_results


def _models_argument(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("provide at least one comma-separated model id")
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model id(s): {', '.join(unknown)}")
    if len(set(models)) != len(models):
        raise argparse.ArgumentTypeError("model ids must not be repeated")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="validate poems and write a deterministic task manifest")
    prepare.add_argument("--data", default=str(DATA_PATH))
    prepare.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    prepare.add_argument("--split", choices=("all", "dev", "test"), default="test")
    prepare.add_argument("--limit-poems", type=int)
    prepare.add_argument("--repetitions", type=int, default=3)
    prepare.add_argument("--models", type=_models_argument, default=MODELS)
    prepare.add_argument("--seed", type=int, default=SHUFFLE_SEED)

    run = commands.add_parser("run", help="run or resume generation from a manifest")
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    run.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    run.add_argument("--split", choices=("dev", "test"))
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=2)

    judge = commands.add_parser("judge", help="prepare a blind batch plan and score completed translations")
    judge.add_argument("--variant", choices=("primary", "secondary", "reversed"), default="primary")
    judge.add_argument("--split", choices=("dev", "test"), default="test")
    judge.add_argument("--judge-model", choices=MODELS)
    judge.add_argument("--generation-dir", default=str(DEFAULT_RESULTS_DIR))
    judge.add_argument("--generation-manifest")
    judge.add_argument("--data", default=str(DATA_PATH))
    judge.add_argument("--results-dir", default=str(DEFAULT_JUDGE_RESULTS_DIR))
    judge.add_argument("--manifest")
    judge.add_argument("--batch-size", type=int, choices=(3, 6), default=BATCH_SIZE)
    judge.add_argument("--limit-batches", type=int)
    judge.add_argument("--sample-batches", type=int)
    judge.add_argument("--sample-seed", type=int, default=SHUFFLE_SEED)
    judge.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=2)
    judge.add_argument("--workers", type=int, default=1)

    summarize = commands.add_parser("summarize", help="aggregate scores and generation diagnostics at poem level")
    summarize.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    summarize.add_argument("--generation-dir", default=str(DEFAULT_RESULTS_DIR))
    summarize.add_argument("--judge-results-dir", default=str(DEFAULT_JUDGE_RESULTS_DIR))
    summarize.add_argument("--output-dir", default=str(DEFAULT_SUMMARY_DIR))
    summarize.add_argument("--bootstrap-seed", type=int, default=20261025)
    summarize.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_manifest(
            data_path=Path(args.data),
            manifest_path=Path(args.manifest),
            split=args.split,
            limit_poems=args.limit_poems,
            seed=args.seed,
            repetitions=args.repetitions,
            models=args.models,
        )
        print(json.dumps({"manifest": args.manifest, "task_count": len(manifest["tasks"])}, ensure_ascii=False))
        return 0
    if args.command == "judge":
        results_dir = Path(args.results_dir)
        manifest_path = Path(args.manifest) if args.manifest else results_dir / f"{args.variant}_{args.split}_manifest.json"
        plan = build_judge_plan(
            variant=args.variant,
            split=args.split,
            generation_results_dir=Path(args.generation_dir),
            generation_manifest_path=Path(args.generation_manifest) if args.generation_manifest else None,
            data_path=Path(args.data),
            judge_model=args.judge_model,
            batch_size=args.batch_size,
            seed=args.sample_seed,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = run_judging(
            manifest_path,
            results_dir,
            limit_batches=args.limit_batches,
            sample_batches=args.sample_batches,
            sample_seed=args.sample_seed,
            max_retries=args.max_retries,
            workers=args.workers,
        )
        counts["manifest"] = str(manifest_path)
        print(json.dumps(counts, ensure_ascii=False))
        return 1 if counts["failed"] or counts["unavailable"] else 0
    if args.command == "summarize":
        summary = summarize_results(
            manifest_path=Path(args.manifest),
            generation_results_dir=Path(args.generation_dir),
            judging_results_dir=Path(args.judge_results_dir),
            output_dir=Path(args.output_dir),
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    counts = run_generation(
        manifest_path=Path(args.manifest),
        results_dir=Path(args.results_dir),
        split=args.split,
        workers=args.workers,
        max_retries=args.max_retries,
    )
    print(json.dumps(counts, ensure_ascii=False))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
