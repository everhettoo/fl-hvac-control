import unittest
import numpy as np
import mylibs.membership_functions as mf

class TestIncreasingMembershipFunction(unittest.TestCase):

    def test_returns_zero_when_x_less_than_or_equal_to_a(self):
        self.assertEqual(mf.inc(5, 10, 20), 0)
        self.assertEqual(mf.inc(10, 10, 20), 0)
        self.assertEqual(mf.inc(-100, 0, 100), 0)

    def test_returns_one_when_x_greater_than_or_equal_to_b(self):
        self.assertEqual(mf.inc(25, 10, 20), 1)
        self.assertEqual(mf.inc(20, 10, 20), 1)
        self.assertEqual(mf.inc(1000, 0, 100), 1)

    def test_returns_interpolated_value_between_a_and_b(self):
        self.assertAlmostEqual(mf.inc(15, 10, 20), 0.5)
        self.assertAlmostEqual(mf.inc(12, 10, 20), 0.2)
        self.assertAlmostEqual(mf.inc(18, 10, 20), 0.8)

    def test_returns_value_in_zero_one_range_for_middle_point(self):
        result = mf.inc(50, 0, 100)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)
        self.assertAlmostEqual(result, 0.5)

    def test_handles_negative_ranges(self):
        self.assertEqual(mf.inc(-15, -10, 0), 0)
        self.assertAlmostEqual(mf.inc(-5, -10, 0), 0.5)
        self.assertEqual(mf.inc(0, -10, 0), 1)

    def test_handles_decimal_values(self):
        self.assertAlmostEqual(mf.inc(2.5, 1.0, 4.0), 0.5)
        self.assertAlmostEqual(mf.inc(1.5, 1.0, 4.0), 1/6, places=5)

    def test_handles_very_small_interval(self):
        result = mf.inc(1.0001, 1.0, 1.0002)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_handles_very_large_values(self):
        self.assertEqual(mf.inc(1000000, 500000, 1000000), 1)
        self.assertEqual(mf.inc(499999, 500000, 1000000), 0)
        self.assertAlmostEqual(mf.inc(750000, 500000, 1000000), 0.5)

    def test_handles_a_equals_b_boundary_case(self):
        self.assertEqual(mf.inc(10, 10, 10), 0)
        self.assertEqual(mf.inc(10.0001, 10, 10), 1)

    def test_handles_x_as_numpy_array(self):
        x_array = np.array([5, 10, 15, 20, 25])
        result = np.array([mf.inc(val, 10, 20) for val in x_array])
        expected = np.array([0, 0, 0.5, 1, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_float_precision_at_boundaries(self):
        a, b = 10.0, 20.0
        self.assertAlmostEqual(mf.inc(a - 0.0001, a, b), 0)
        self.assertAlmostEqual(mf.inc(b + 0.0001, a, b), 1)

    def test_produces_continuous_monotonic_increasing_output(self):
        a, b = 10, 20
        x_values = np.linspace(5, 25, 100)
        y_values = [mf.inc(x, a, b) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertLessEqual(y_values[i], y_values[i + 1])

    def test_handles_single_point_interval_a_equals_b(self):
        self.assertEqual(mf.inc(9.99, 10, 10), 0)
        self.assertEqual(mf.inc(10.00, 10, 10), 0)
        self.assertEqual(mf.inc(10.01, 10, 10), 1)

    def test_returns_exactly_half_at_midpoint(self):
        a, b = 0, 10
        midpoint = (a + b) / 2
        self.assertAlmostEqual(mf.inc(midpoint, a, b), 0.5)

    def test_handles_co2_sensor_range(self):
        a, b = 400, 1500
        self.assertEqual(mf.inc(350, a, b), 0)
        self.assertEqual(mf.inc(1500, a, b), 1)
        self.assertAlmostEqual(mf.inc(950, a, b), (950 - 400) / (1500 - 400))

    def test_handles_temperature_sensor_range(self):
        a, b = 18, 30
        self.assertEqual(mf.inc(10, a, b), 0)
        self.assertEqual(mf.inc(30, a, b), 1)
        self.assertAlmostEqual(mf.inc(24, a, b), 0.5)

    def test_handles_humidity_sensor_range(self):
        a, b = 30, 80
        self.assertEqual(mf.inc(25, a, b), 0)
        self.assertEqual(mf.inc(80, a, b), 1)
        self.assertAlmostEqual(mf.inc(55, a, b), 0.5)

    def test_returns_scalar_for_scalar_input(self):
        result = mf.inc(15, 10, 20)
        self.assertIsInstance(result, (int, float))

    def test_maintains_mathematical_linearity_in_middle_region(self):
        a, b = 10, 20
        x1, x2 = 12, 14
        y1 = mf.inc(x1, a, b)
        y2 = mf.inc(x2, a, b)

        slope = (y2 - y1) / (x2 - x1)
        expected_slope = 1 / (b - a)
        self.assertAlmostEqual(slope, expected_slope)

    def test_handles_inverted_range_gracefully(self):
        result = mf.inc(15, 20, 10)
        self.assertIsNotNone(result)

    def test_handles_zero_crossing_interval(self):
        a, b = -5, 5
        self.assertEqual(mf.inc(-10, a, b), 0)
        self.assertEqual(mf.inc(5, a, b), 1)
        self.assertAlmostEqual(mf.inc(0, a, b), 0.5)

    def test_boundary_value_a_returns_zero(self):
        for a in [0, 100, -50, 3.14]:
            self.assertEqual(mf.inc(a, a, a + 10), 0)

    def test_boundary_value_b_returns_one(self):
        for b in [10, 100, -50, 3.14]:
            self.assertEqual(mf.inc(b, b - 10, b), 1)

    def test_handles_consecutive_calls_with_same_parameters(self):
        result1 = mf.inc(15, 10, 20)
        result2 = mf.inc(15, 10, 20)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()

