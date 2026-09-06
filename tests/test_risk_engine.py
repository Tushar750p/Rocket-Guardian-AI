import unittest

import pandas as pd

from src.risk_engine_v14 import analyze, risk_level, sensor_risk


class RiskEngineTests(unittest.TestCase):
    def make_data(self):
        return pd.DataFrame(
            {
                "time_s": [0.0, 1.0],
                "pressure_kpa_z": [0.0, 8.0],
                "temperature_k_z": [0.0, 2.0],
                "vibration_g_z": [0.0, 10.0],
                "thrust_n_z": [0.0, 1.0],
                "ai_anomaly": [0, 1],
                "anomaly": [0, 1],
            }
        )

    def test_sensor_risk_is_bounded(self):
        values = sensor_risk(pd.Series([-100.0, 0.0, 100.0]))
        self.assertTrue((values >= 0).all())
        self.assertTrue((values <= 100).all())
        self.assertAlmostEqual(float(values.iloc[1]), 0.0, places=6)

    def test_risk_levels(self):
        self.assertEqual(risk_level(20), "NORMAL")
        self.assertEqual(risk_level(45), "WARNING")
        self.assertEqual(risk_level(75), "CRITICAL")

    def test_analyze_adds_risk_columns_and_primary_sensor(self):
        result = analyze(self.make_data())
        for column in (
            "pressure_risk",
            "temperature_risk",
            "vibration_risk",
            "thrust_risk",
            "overall_risk",
            "risk_level",
            "primary_risk_sensor",
            "elevated_sensor_count",
            "risk_explanation",
        ):
            self.assertIn(column, result.columns)

        peak = result.iloc[-1]
        self.assertEqual(peak["primary_risk_sensor"], "Vibration")
        self.assertEqual(peak["risk_level"], "CRITICAL")
        self.assertGreaterEqual(float(peak["overall_risk"]), 75.0)
        self.assertEqual(int(peak["elevated_sensor_count"]), 2)


if __name__ == "__main__":
    unittest.main()
