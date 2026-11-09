"""Reusable reader, swim windows and estimators from the project's 2026-11-06 replay."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BIN = 30


REQUIRED = {
    "心率": {"F": "心率（次/分）", "G": "类型", "H": "时间"},
    "单次运动": {"F": "运动类型", "G": "开始时间", "H": "结束时间"},
    "日常活动状态": {"F": "活动状态类型", "H": "时间"},
}


def read_workbook(payload, keep_identity=False):
    data, headers = {}, {}
    with ZipFile(BytesIO(payload)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.findall(".//{*}t")) for item in shared
            ]
        links = {
            element.attrib["Id"]: element.attrib["Target"]
            for element in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in book.findall(".//{*}sheet"):
            name = sheet.attrib["name"]
            link_id = next(v for k, v in sheet.attrib.items() if k.endswith("}id"))
            part = links[link_id]
            root = ET.fromstring(archive.read(part.lstrip("/") if part.startswith("/") else "xl/" + part))
            rows = []
            for row in root.findall(".//{*}sheetData/{*}row"):
                values = {}
                for cell in row:
                    column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                    value = cell.findtext("{*}v")
                    if cell.get("t") == "s" and value is not None:
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall(".//{*}t"))
                    if value is not None and value.strip():
                        values[column] = value.strip()
                if values:
                    rows.append(values)
            headers[name], *data[name] = rows
    for sheet, required in REQUIRED.items():
        if not required.items() <= headers[sheet].items():
            raise ValueError(f"Unexpected workbook columns: {sheet}")
    if keep_identity:
        return data
    return {name: [drop_identity(row) for row in rows] for name, rows in data.items()}


def drop_identity(row):
    return {key: value for key, value in row.items() if key not in {"A", "B", "C", "D", "E"}}


@dataclass
class Swim:
    key: str
    start: datetime
    duration: float
    seconds: np.ndarray
    heart: np.ndarray
    points: int
    conflicts: int

    @property
    def date(self):
        return self.start.date().isoformat()


def build_swims(data):
    states = defaultdict(set)
    for row in data["日常活动状态"]:
        states[datetime.fromisoformat(row["H"])].add(row["F"])
    heart = defaultdict(set)
    for row in data["心率"]:
        if row["G"] == "实时心率":
            heart[datetime.fromisoformat(row["H"])].add(float(row["F"]))

    workouts = sorted(data["单次运动"], key=lambda row: row["G"])
    swims, labels = [], Counter()
    for row in workouts:
        start = datetime.fromisoformat(row["G"])
        end = datetime.fromisoformat(row["H"])
        matched = states[start]
        activity = row.get("F") or (next(iter(matched)) if len(matched) == 1 else "未匹配")
        labels[activity] += 1
        if activity != "游泳":
            continue
        duration = (end - start).total_seconds()
        windows = [[] for _ in range(int(duration // BIN))]
        points = conflicts = 0
        for instant, values in heart.items():
            if not start <= instant < end:
                continue
            if len(values) != 1:
                conflicts += 1
                continue
            value = next(iter(values))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid heart-rate input")
            points += 1
            position = int((instant - start).total_seconds() // BIN)
            if position < len(windows):
                windows[position].append(value)
        medians = np.array([np.median(window) if len(window) >= 2 else np.nan for window in windows])
        swims.append(Swim(f"S{len(swims) + 1:02}", start, duration,
                          BIN * np.arange(1, len(windows) + 1), medians, points, conflicts))
    return swims, dict(labels)


def cases(swims, horizon, include_unobserved=False):
    out = []
    step = horizon // BIN
    for swim in swims:
        for i in range(2, len(swim.heart) - step):
            j = i + step
            past = swim.heart[i - 2:i + 1]
            if np.isfinite(past).all() and (include_unobserved or np.isfinite(swim.heart[j])):
                earlier, previous, current = past
                features = [current, previous, earlier, np.mean(past), current - earlier,
                            np.log1p(swim.seconds[i] / 60)]
                out.append((swim, i, j, features, swim.heart[j]))
    return out


def values(records):
    return np.asarray([r[4] for r in records])


def matrix(records):
    return np.asarray([r[3] for r in records])


def weights(records):
    counts = Counter(row[0].key for row in records)
    return len(records) / len(counts) / np.array([counts[row[0].key] for row in records])


def fit_models(records):
    x, y, w = matrix(records), values(records), weights(records)
    mean = float(np.average(y, weights=w))
    delta = mean - x[:, 0]
    step = float(np.clip(np.sum(w * delta * (y - x[:, 0])) / np.sum(w * delta**2), 0, 1))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=15.0))
    ridge.fit(x, y, ridge__sample_weight=w)
    boosted = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=100, max_leaf_nodes=7, min_samples_leaf=25,
        l2_regularization=2.0, early_stopping=False, random_state=0,
    ).fit(x, y, sample_weight=w)
    return {
        "persistence": lambda z: z[:, 0],
        "history_mean": lambda z: np.full(len(z), mean),
        "mean_reversion": lambda z: z[:, 0] + step * (mean - z[:, 0]),
        "ridge": ridge.predict,
        "boosted_tree": boosted.predict,
    }


