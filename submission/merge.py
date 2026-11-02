# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check each task{N}/submit/index.csv and merge them into one Kaggle submission.

Run from the repository root:

    uv run submission/merge.py
    uv run submission/merge.py --baseline base   # also report changes vs. another checkout

Exits 1 when any check fails. Prints a Markdown report to stdout.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

TASKS = ("task1", "task2", "task3", "task4", "task5")
TASK4_LABELS = {"7-11K", "fiveN", "freefried", "pube", "roblock"}
SAMPLE = Path("submission/sample_submission.csv")


def task_file(root: Path, task: str) -> Path:
    return root / task / "submit" / "index.csv"


def check_value(task: str, raw: str) -> str | None:
    """Return the normalized value, or None if the value is invalid."""
    value = raw.strip()
    if task in ("task1", "task5"):
        return value if value in {"0", "1"} else None
    if task == "task2":
        return value if value in {"0", "1", "2", "3"} else None
    if task == "task3":
        try:
            number = float(value)
        except ValueError:
            return None
        return value if math.isfinite(number) and number >= 0 else None
    if task == "task4":
        return value if value in TASK4_LABELS else None
    raise ValueError(task)


def read_task(
    path: Path, task: str, ids: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Read one task file. Return id -> value and a list of errors."""
    if not path.is_file():
        return {}, [f"`{path}` is missing"]

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    if header is None or [h.strip() for h in header] != ["id", task]:
        return {}, [
            f"`{path}` header must be `id,{task}`, got `{','.join(header or [])}`"
        ]

    errors: list[str] = []
    values: dict[str, str] = {}
    expected = set(ids)
    bad_values: list[str] = []
    unknown: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for line_no, row in enumerate(rows, start=2):
        if len(row) != 2:
            errors.append(
                f"`{path}` line {line_no}: expected 2 columns, got {len(row)}"
            )
            continue
        row_id, raw = row[0].strip(), row[1]
        if row_id not in expected:
            unknown.append(row_id)
            continue
        if row_id in seen:
            duplicates.append(row_id)
            continue
        seen.add(row_id)
        value = check_value(task, raw)
        if value is None:
            bad_values.append(f"{row_id}={raw!r}")
            continue
        values[row_id] = value

    missing = [i for i in ids if i not in seen]
    for label, items in (
        ("unknown ids", unknown),
        ("duplicate ids", duplicates),
        ("invalid values", bad_values),
        ("missing ids", missing),
    ):
        if items:
            preview = ", ".join(items[:5]) + (" …" if len(items) > 5 else "")
            errors.append(f"`{path}`: {len(items)} {label} ({preview})")
    return values, errors


def warnings_for(task: str, values: dict[str, str]) -> list[str]:
    if task != "task3" or not values:
        return []
    numbers = [float(v) for v in values.values()]
    if 0 < max(numbers) < 25:
        return [
            f"task3 max is {max(numbers):.3g} THB – check that the log transform is inverted"
        ]
    return []


def distribution(task: str, values: dict[str, str]) -> str:
    if task == "task3":
        numbers = sorted(float(v) for v in values.values())
        zeros = sum(n == 0 for n in numbers) / len(numbers)
        median = numbers[len(numbers) // 2]
        return f"zeros {zeros:.1%}, median {median:.6g}, max {numbers[-1]:.6g}"
    counts: dict[str, int] = {}
    for v in values.values():
        counts[v] = counts.get(v, 0) + 1
    return ", ".join(
        f"`{k}` {n:,} ({n / len(values):.1%})" for k, n in sorted(counts.items())
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--out", type=Path, default=Path("submission/submission.csv"))
    parser.add_argument(
        "--baseline", type=Path, help="root of another checkout to diff against"
    )
    args = parser.parse_args()

    with (args.root / SAMPLE).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", *TASKS]:
            print(f"`{SAMPLE}` header is wrong: {reader.fieldnames}", file=sys.stderr)
            return 1
        ids = [row["id"] for row in reader]

    results = {task: read_task(task_file(args.root, task), task, ids) for task in TASKS}
    errors = [e for _, errs in results.values() for e in errs]

    lines = ["## Submission check", "", f"Rows expected: {len(ids):,}", ""]
    lines += [
        "| Task | Status | Distribution | Changed vs. base |",
        "| --- | --- | --- | --- |",
    ]
    warnings: list[str] = []
    for task, (values, errs) in results.items():
        changed = "–"
        if args.baseline and not errs:
            base_values, base_errs = read_task(
                task_file(args.baseline, task), task, ids
            )
            if not base_errs:
                n = sum(values[i] != base_values[i] for i in ids)
                changed = f"{n:,} ({n / len(ids):.1%})"
            else:
                changed = "n/a"
        status = "❌" if errs else "✅"
        dist = distribution(task, values) if not errs else "–"
        lines.append(f"| {task} | {status} | {dist} | {changed} |")
        if not errs:
            warnings += warnings_for(task, values)

    if warnings:
        lines += ["", "### Warnings", "", *(f"- {w}" for w in warnings)]
    if errors:
        lines += ["", "### Errors", "", *(f"- {e}" for e in errors)]
    print("\n".join(lines))
    if errors:
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", *TASKS])
        for i in ids:
            writer.writerow([i, *(results[t][0][i] for t in TASKS)])
    print(f"\nWrote `{args.out}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
