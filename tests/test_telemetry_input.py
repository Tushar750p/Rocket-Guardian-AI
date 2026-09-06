import io
import unittest

import pandas as pd

from src.telemetry_input import load_telemetry_csv


class TelemetryInputTests(unittest.TestCase):
    def make_csv(self, **overrides):
        data = {
            "time_s": [2, 1, 3],
            "phase": ["steady", "startup", "shutdown"],
            "pressure_kpa": [100, 101, 99],
            "temperature_k": [300, 301, 299],
            "vibration_g": [1.0, 1.1, 0.9],
            "thrust_n": [500, 510, 490],
        }
        data.update(overrides)
        return pd.DataFrame(data).to_csv(index=False)

    def test_valid_csv_loads_and_is_sorted(self):
        result = load_telemetry_csv(io.StringIO(self.make_csv()))
        self.assertEqual(len(result), 3)
        self.assertEqual(result["time_s"].tolist(), [1, 2, 3])
        self.assertEqual(result["phase"].tolist(), ["startup", "steady", "shutdown"])

    def test_missing_required_column_raises(self):
        csv_text = self.make_csv().replace("thrust_n", "missing_thrust", 1)
        with self.assertRaisesRegex(ValueError, "Missing required columns: thrust_n"):
            load_telemetry_csv(io.StringIO(csv_text))

    def test_non_numeric_sensor_value_raises(self):
        csv_text = self.make_csv(vibration_g=[1.0, "bad", 0.9])
        with self.assertRaisesRegex(ValueError, "Column 'vibration_g' contains invalid"):
            load_telemetry_csv(io.StringIO(csv_text))

    def test_empty_csv_raises(self):
        empty = "time_s,phase,pressure_kpa,temperature_k,vibration_g,thrust_n\n"
        with self.assertRaisesRegex(ValueError, "Telemetry file is empty"):
            load_telemetry_csv(io.StringIO(empty))

    def test_missing_phase_raises(self):
        csv_text = self.make_csv(phase=["startup", None, "shutdown"])
        with self.assertRaisesRegex(ValueError, "Column 'phase' contains missing values"):
            load_telemetry_csv(io.StringIO(csv_text))


if __name__ == "__main__":
    unittest.main()
