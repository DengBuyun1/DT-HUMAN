import unittest
from datetime import datetime, timedelta

import numpy as np

from v3.aggregate import sensor_grid, sleep_days
from v3.data_quality import clean_measurements
from v3.daily import evaluate_series
from v3.heart import replay_swim
from v3.heat import fit_heat
from v3.evaluate import stats
import swim


class ContractTests(unittest.TestCase):
    def test_conflict_dedup_and_grid(self):
        day = datetime(2000,1,2)
        measurements = [(day + timedelta(minutes=i), v, str(j))
                        for j, (i, v) in enumerate([(0, 60), (0, 60), (1, 62), (1, 85), (5, 80), (10, 90)])]
        points, audit = clean_measurements(measurements, positive=True)
        self.assertEqual(sorted(x["status"] for x in audit),
                         ["conflict", "conflict", "duplicate", "kept", "kept", "kept"])
        hourly, _, _ = sensor_grid(points, [day], 3, 1, 1)
        self.assertEqual(hourly[day].value, 80)
        self.assertEqual(hourly[day].count, 3)
        self.assertEqual(len(hourly), 1)

    def test_nap_sum_and_median(self):
        rows = [{"F":"2000-01-01T10:00:00","G":"2000-01-01T10:20:00","O":"零星小睡","H":"20"},
                {"F":"2000-01-01T14:00:00","G":"2000-01-01T14:30:00","O":"零星小睡","H":"30"}]
        values, _, _ = sleep_days(rows)
        day = datetime(2000, 1, 1)
        self.assertEqual(values["小睡总时长"][day].value, 50)
        self.assertEqual(values["单次小睡时长中位数"][day].value, 25)

    def test_hourly_dates(self):
        origin = datetime(2000,1,2,23)
        rows = [dict(date=(origin+timedelta(hours=i)).date().isoformat(), observed=80, prediction=81,
                     session=None, prediction_id=str(i)) for i in range(25)]
        self.assertEqual(stats(rows)["n_dates"], 2)
        self.assertEqual(stats(rows)["n"], 25)

    def test_daily_no_leak_and_no_gap_pairs(self):
        from v3.aggregate import Point
        from v3.splits import Window
        from v3.run import CONFIG
        day = datetime(2000,1,2)
        grid = [day+timedelta(days=i) for i in range(18)]
        target = {t:Point(60+i%6,1,(str(i),)) for i,t in enumerate(grid) if i != 4}
        window = Window("V3-F50","demo",grid[:12],grid[12:])
        kwargs = dict(target_id="test", window=window, step=timedelta(days=1), config=CONFIG)
        first, fit = evaluate_series(points=target, **kwargs)
        target[grid[12]]=Point(400,1,("future",))
        second, other_fit = evaluate_series(points=target, **kwargs)
        self.assertEqual(fit,other_fit)
        self.assertEqual(fit[0]["train_pairs"],9)
        for a,b in zip(first,second):
            if a["information_mode"]=="open_loop" or a["target_start"]==grid[12]:
                self.assertEqual(a["prediction"],b["prediction"])

    def test_swim_future_missing_and_tail(self):
        from v3.heart import make_predictors
        start = datetime(2000,1,2,12)
        s = swim.Swim("S00",start,300,np.arange(30,301,30),np.arange(100.,110.),10,0)
        models = make_predictors([],120,3,15)[0]
        rows = replay_swim(s,120,models,{"protocol_id":"V3-F50","train_end":start,"segment_id":"swim",
                                         "model_version":"test"})
        s.heart[6] = np.nan
        other = replay_swim(s,120,models,{"protocol_id":"V3-F50","train_end":start,"segment_id":"swim",
                                          "model_version":"test"})
        before = start + timedelta(seconds=210)
        self.assertEqual([(x["issued_at"],x["prediction"]) for x in rows if x["origin"]<before],
                         [(x["issued_at"],x["prediction"]) for x in other if x["origin"]<before])
        self.assertTrue(any(x["status"]=="outside_record" and x["prediction"] is not None for x in rows))

    def test_heat_zero_intercept(self):
        coeff = fit_heat([{"minutes":10,"observed":50},{"minutes":20,"observed":130}])
        self.assertEqual(coeff["E-R0"][0],0)
        self.assertEqual(coeff["E-R2"][0],0)
        self.assertAlmostEqual(coeff["E-R2"][1],6.2)


if __name__ == "__main__":
    unittest.main()
