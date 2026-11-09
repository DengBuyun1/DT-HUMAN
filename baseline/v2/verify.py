"""Check the time and observation boundaries in a completed local replay."""
import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


def verify(run):
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        assert sha256((run / name).read_bytes()).hexdigest() == expected, name
    with (run / "swim_heart.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["id"]: r for r in csv.DictReader(stream)}
    con = sqlite3.connect(run / "heart_replay.sqlite")
    try:
        entries = list(con.execute(
            "SELECT distribution, registered_at, target_time, outcome, scored_at, withdrawn_at FROM predictions"))
    finally:
        con.close()
    assert len(entries) == sum(bool(row["selected"]) for row in rows.values())
    matched = set()
    scored = 0
    for distribution, origin, target, outcome, score, withdrawn in entries:
        metadata = json.loads(distribution)
        key = metadata["window_id"]
        row = rows[key]
        assert key not in matched
        matched.add(key)
        assert row["origin"] == origin == metadata["input_cutoff"]
        assert row["available"] == target and row["fit_through"] < row["date"]
        assert (datetime.fromisoformat(target) - datetime.fromisoformat(origin)).total_seconds() == int(row["horizon"])
        assert float(row["selected"]) == metadata["value"]
        if row["status"] == "scored":
            assert score == target and outcome == float(row["observed"]) and withdrawn is None
            scored += 1
        elif row["status"] == "missing_target":
            assert score is None and withdrawn == target and not row["observed"]
        else:
            assert row["status"] == "outside_recording" and score is None and withdrawn is not None
    assert scored == report["heart_ledger"]["scored"]
    assert len(matched) == report["heart_ledger"]["issued"]
    return {"issued": len(matched), "scored": scored, "withdrawn": len(matched) - scored,
            "verified_artifacts": len(manifest["artifact_sha256"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))
