"""One measurement-cleaning path for daily, hourly and swim observations."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import isfinite

TZ = timezone(timedelta(hours=8), "Asia/Shanghai")
SENSORS = {
    "实时心率": ("心率", "H", "次/分"),
    "静息心率": ("心率", "H", "次/分"),
    "血氧": ("血氧饱和度", "G", "%"),
    "压力": ("压力", "G", "分"),
}


def local_time(value):
    stamp = datetime.fromisoformat(value)
    return stamp.astimezone(TZ).replace(tzinfo=None) if stamp.tzinfo else stamp


def clean_measurements(records, positive=False):
    groups = defaultdict(list)
    audit = []
    for stamp, value, row_id in records:
        row = dict(raw_row_id=row_id, timestamp=stamp, value=value, cleaned_record_id=None)
        if stamp is None or value is None or not isfinite(value) or (positive and value <= 0):
            audit.append(dict(row, status="invalid"))
        else:
            groups[stamp].append(row)
    points = {}
    for stamp, group in sorted(groups.items()):
        if len({row["value"] for row in group}) > 1:
            audit.extend(dict(row, status="conflict") for row in group)
            continue
        first = group[0]
        clean_id = "clean:" + first["raw_row_id"]
        points[stamp] = (first["value"], clean_id)
        audit.extend(dict(row, status="kept" if i == 0 else "duplicate",
                          cleaned_record_id=clean_id) for i,row in enumerate(group))
    return points, audit


def clean_sensors(data):
    cleaned, audit = {}, []
    for metric, (sheet, time_column, _) in SENSORS.items():
        records = []
        for i, row in enumerate(data[sheet], 1):
            if sheet == "心率" and row.get("G") != metric:
                continue
            try:
                stamp = local_time(row[time_column])
            except (KeyError, ValueError):
                stamp = None
            try:
                value = float(row["F"])
                if not isfinite(value):
                    value = None
            except (KeyError, ValueError):
                value = None
            records.append((stamp, value, f"{sheet}:{i}"))
        cleaned[metric], log = clean_measurements(records, metric in ("实时心率", "静息心率"))
        audit.extend(dict(row, metric=metric) for row in log)
    return cleaned, audit
