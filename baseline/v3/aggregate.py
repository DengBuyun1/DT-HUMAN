"""Calendar-aligned summaries and explicit coverage."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from statistics import median

from v3.data_quality import local_time

DAY = timedelta(days=1)
HOUR = timedelta(hours=1)
FIVE = timedelta(minutes=5)


@dataclass(frozen=True)
class Point:
    value: float
    count: int
    sources: tuple[str, ...]


def floor_day(stamp):
    return stamp.replace(hour=0, minute=0, second=0, microsecond=0)


def floor_hour(stamp):
    return stamp.replace(minute=0, second=0, microsecond=0)


def summarized(values):
    return Point(float(median(p.value for p in values)), len(values),
                 tuple(s for p in values for s in p.sources))


def sensor_grid(points, days, min_bins, min_hours, min_blocks):
    bins = defaultdict(list)
    raw_hours = defaultdict(int)
    for stamp, (value, record_id) in sorted(points.items()):
        hour = floor_hour(stamp)
        bins[hour + FIVE*(stamp.minute//5)].append(Point(value, 1, (record_id,)))
        raw_hours[hour] += 1
    hour_bins = defaultdict(list)
    for stamp, readings in sorted(bins.items()):
        hour_bins[floor_hour(stamp)].append(summarized(readings))
    hourly, daily, coverage = {}, {}, []
    for day in days:
        available = []
        for i in range(24):
            hour = day + i*HOUR
            blocks = hour_bins[hour]
            ok = len(blocks) >= min_bins
            if ok:
                hourly[hour] = summarized(blocks)
                available.append(hour)
            coverage.append(dict(slot=hour, resolution="hour", raw_unique=raw_hours[hour],
                                 valid_5min_bins=len(blocks), valid_hours=int(ok), blocks_6h=None,
                                 eligible=ok, hour_mask=None))
        n_blocks = len({t.hour//6 for t in available})
        ok = len(available) >= min_hours and n_blocks >= min_blocks
        if ok:
            daily[day] = summarized([hourly[t] for t in available])
        coverage.append(dict(slot=day, resolution="day", raw_unique=sum(raw_hours[day+i*HOUR] for i in range(24)),
                             valid_5min_bins=sum(len(hour_bins[day+i*HOUR]) for i in range(24)),
                             valid_hours=len(available), blocks_6h=n_blocks, eligible=ok,
                             hour_mask="".join("1" if day+i*HOUR in hourly else "0" for i in range(24))))
    return hourly, daily, coverage


def sleep_days(rows):
    events, audit = [], []
    seen = set()
    for i, row in enumerate(rows, 1):
        row_id = f"睡眠:{i}"
        kind = row.get("O")
        try:
            start, end, minutes = local_time(row["F"]), local_time(row["G"]), float(row["H"])
            if start >= end or not isfinite(minutes) or minutes <= 0 or kind not in ("科学睡眠","零星小睡"):
                raise ValueError("invalid event")
        except (KeyError, ValueError):
            audit.append(dict(raw_row_id=row_id, status="invalid"))
            continue
        key = (start,end,minutes,kind)
        if key in seen:
            audit.append(dict(raw_row_id=row_id, status="duplicate"))
            continue
        seen.add(key)
        events.append(dict(raw_row_id=row_id, start=start, end=end, minutes=minutes, kind=kind))
    by_kind = defaultdict(list)
    for event in events:
        others = [x for x in events if x is not event and x["start"] < event["end"] and x["end"] > event["start"]]
        status = "overlap" if others else "kept"
        event["status"] = status
        audit.append(dict(raw_row_id=event["raw_row_id"], status=status))
        if not others:
            by_kind[event["kind"], floor_day(event["end"])].append(
                Point(event["minutes"], 1, (event["raw_row_id"],)))
    result = {"主睡眠时长":{}, "小睡总时长":{}, "单次小睡时长中位数":{}}
    for (kind, day), points in by_kind.items():
        point = summarized(points)
        if kind == "科学睡眠":
            result["主睡眠时长"][day] = point
        else:
            result["小睡总时长"][day] = Point(sum(p.value for p in points), point.count, point.sources)
            result["单次小睡时长中位数"][day] = point
    return result, events, audit


def day_values(records):
    values = defaultdict(list)
    for stamp, value, row_id in records:
        values[floor_day(stamp)].append(Point(float(value), 1, (row_id,)))
    return {day:summarized(points) for day,points in sorted(values.items())}
