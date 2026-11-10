"""Verify saved files and time ordering for the experiment archive."""
import argparse
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_rows(rows):
    keys = set()
    for row in rows:
        key = row["prediction_id"]
        assert key not in keys, key
        keys.add(key)
        issued = row["issued_at"]
        target = row["target_end"]
        if issued is not None:
            assert row["train_end"] <= issued <= target
            assert row["input_end"] is None or row["input_end"] <= issued
            assert row["prediction"] is not None
            if row["information_mode"]!="conditional_duration":
                assert issued < target
                assert issued<=row["target_start"]
        if row["status"]=="outside_record":
            assert row["prediction"] is not None and row["observed"] is None
        assert row["available_at"]>=target
    return len(keys)


def verify_directory(folder):
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="completed"
    for name, checksum in manifest["artifact_sha256"].items():
        assert digest(folder/name)==checksum, name
    return manifest


def verify_reproduction(a,b):
    left,right=verify_directory(a),verify_directory(b)
    for key in ("source_sha256","plan_sha256","code_sha256","environment"):
        assert left[key]==right[key],key
    names=("protocol.json","metrics.json","predictions.csv","calibration.csv","split_manifest.csv",
           "model_parameters.json","decision_table.csv","per_date_metrics.csv","per_session_metrics.csv",
           "coverage.csv","sensitivity_results.csv")
    for name in names:
        assert left["artifact_sha256"][name]==right["artifact_sha256"][name],name
    return list(names)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--compare",type=Path)
    args=parser.parse_args()
    manifest=verify_directory(args.run)
    print("PASS files",len(manifest["artifact_sha256"]))
    if args.compare:
        print("PASS repeat",", ".join(verify_reproduction(args.run,args.compare)))


if __name__=="__main__":
    main()
