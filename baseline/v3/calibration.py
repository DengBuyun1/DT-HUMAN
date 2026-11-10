"""Independent online calibration with only earlier completed dates."""
from collections import defaultdict

import numpy as np

from v3.evaluate import valid


def calibrate(rows,config):
    history = defaultdict(list)
    predictions = []
    for day in sorted({r["date"] for r in rows}):
        current = [r for r in rows if r["date"]==day]
        for row in current:
            if row["prediction"] is None:
                continue
            key = row["target_id"],row["model_id"]
            previous = history[key]
            dates = defaultdict(lambda:defaultdict(list))
            for item in previous:
                dates[item["date"]][item["session"]].append(abs(item["prediction"]-item["observed"]))
            q = None
            if len(dates)>=config["band_min_dates"]:
                values = sorted((error,1/len(dates)/len(sessions)/len(group))
                                for sessions in dates.values() for group in sessions.values() for error in group)
                cumulative = np.cumsum([weight for _,weight in values])
                index = int(np.searchsorted(cumulative,config["band_quantile"]))
                q = values[index][0]
            predictions.append(dict(protocol_id="V3-CAL",source_prediction_id=row["prediction_id"],
                                    target_id=row["target_id"],model_id=row["model_id"],date=day,
                                    observed=row["observed"],prediction=row["prediction"],historical_dates=len(dates),
                                    historical_date_ids=tuple(sorted(dates)),
                                    lower=row["prediction"]-q if q is not None else None,
                                    upper=row["prediction"]+q if q is not None else None,
                                    status="enabled" if q is not None else "insufficient_history"))
        for item in valid(current):
            history[item["target_id"],item["model_id"]].append(item)
    return predictions
