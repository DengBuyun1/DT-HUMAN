"""Split calendars before fitting; keep gaps explicit."""
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class Window:
    protocol_id: str
    segment_id: str
    train: list
    test: list
    aliases: tuple = ()


def runs(points, step):
    result = []
    for stamp in sorted(points):
        if not result or stamp-result[-1][-1] != step:
            result.append([])
        result[-1].append(stamp)
    return sorted(result, key=lambda run: (-len(run),run[0]))


def windows(target_id, points, grid, step, cutoff, fixed):
    ordered = runs(points, step)
    segment = ordered[0] if ordered else []
    half = len(segment)//2
    continuous = Window("V3-C50",target_id+":longest",segment[:half],segment[half:])
    if not fixed:
        return [continuous]
    train = [t for t in grid if t < cutoff]
    test = [t for t in grid if t >= cutoff]
    if (train,test)==(continuous.train,continuous.test):
        return [Window("V3-F50",target_id+":full",train,test,("V3-C50",))]
    return [Window("V3-F50",target_id+":full",train,test),continuous]


def prior_sessions(sessions, boundary):
    """Use all available prior dates, excluding unresolved cross-boundary sessions."""
    return [s for s in sessions if s.start.date() < boundary.date()
            and s.start + timedelta(seconds=s.duration) <= boundary]
