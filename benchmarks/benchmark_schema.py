from dataclasses import dataclass
import csv
import math


DTYPES = {"fp16", "bf16", "fp32"}
RESULT_FIELDS = (
    "suite", "variant", "dtype", "rows", "hidden", "elements",
    "device", "soc", "vector_cores", "warmup", "iterations",
    "latency_us_min", "latency_us_p50", "latency_us_p90",
    "logical_bytes", "effective_gbps", "gelements_per_s",
    "checksum", "status",
)


@dataclass(frozen=True)
class BenchmarkCase:
    rows: int
    hidden: int
    dtypes: tuple[str, ...]
    purpose: str


def load_operator_cases(path):
    cases = []
    with open(path, newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            rows, hidden = int(row["rows"]), int(row["hidden"])
            dtypes = tuple(row["dtypes"].split("|"))
            problems = []
            if rows <= 0:
                problems.append("rows must be positive")
            if not 64 <= hidden <= 32768:
                problems.append("hidden must be in [64, 32768]")
            if not dtypes or not set(dtypes) <= DTYPES:
                problems.append("dtype must be fp16, bf16, or fp32")
            if problems:
                raise ValueError(f"line {line}: {'; '.join(problems)}")
            cases.append(BenchmarkCase(rows, hidden, dtypes, row["purpose"]))
    return cases


def validate_result_rows(rows):
    identities = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in RESULT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index}: missing columns {missing}")
        if int(row["iterations"]) <= 0:
            raise ValueError(f"row {index}: iterations must be positive")
        for field in ("latency_us_min", "latency_us_p50", "latency_us_p90"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"row {index}: {field} must be finite")
        identity = tuple(row[field] for field in
                         ("suite", "variant", "dtype", "rows", "hidden"))
        if identity in identities:
            raise ValueError(f"row {index}: duplicate identity {identity}")
        identities.add(identity)
