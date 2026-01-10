import unittest

import numpy as np

import mylibs.membership_functions as mf


class TestDecreasingMembershipFunction(unittest.TestCase):

    def test_returns_one_when_x_less_than_or_equal_to_a(self):
        self.assertEqual(mf.dec(5, 10, 20), 1)
        self.assertEqual(mf.dec(10, 10, 20), 1)
        self.assertEqual(mf.dec(-100, 0, 100), 1)

    def test_returns_zero_when_x_greater_than_or_equal_to_b(self):
        self.assertEqual(mf.dec(25, 10, 20), 0)
        self.assertEqual(mf.dec(20, 10, 20), 0)
        self.assertEqual(mf.dec(1000, 0, 100), 0)

    def test_returns_interpolated_value_between_a_and_b(self):
        self.assertAlmostEqual(mf.dec(15, 10, 20), 0.5)
        self.assertAlmostEqual(mf.dec(12, 10, 20), 0.8)
        self.assertAlmostEqual(mf.dec(18, 10, 20), 0.2)

    def test_returns_value_in_zero_one_range_for_middle_point(self):
        result = mf.dec(50, 0, 100)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)
        self.assertAlmostEqual(result, 0.5)

    def test_handles_negative_ranges(self):
        self.assertEqual(mf.dec(-15, -10, 0), 1)
        self.assertAlmostEqual(mf.dec(-5, -10, 0), 0.5)
        self.assertEqual(mf.dec(0, -10, 0), 0)

    def test_handles_decimal_values(self):
        self.assertAlmostEqual(mf.dec(2.5, 1.0, 4.0), 0.5)
        self.assertAlmostEqual(mf.dec(1.5, 1.0, 4.0), 5 / 6, places=5)

    def test_handles_very_small_interval(self):
        result = mf.dec(1.0001, 1.0, 1.0002)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_handles_very_large_values(self):
        self.assertEqual(mf.dec(1000000, 500000, 1000000), 0)
        self.assertEqual(mf.dec(499999, 500000, 1000000), 1)
        self.assertAlmostEqual(mf.dec(750000, 500000, 1000000), 0.5)

    def test_handles_a_equals_b_boundary_case(self):
        self.assertEqual(mf.dec(10, 10, 10), 1)
        self.assertEqual(mf.dec(10.0001, 10, 10), 0)

    def test_handles_x_as_numpy_array(self):
        x_array = np.array([5, 10, 15, 20, 25])
        result = np.array([mf.dec(val, 10, 20) for val in x_array])
        expected = np.array([1, 1, 0.5, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_float_precision_at_boundaries(self):
        a, b = 10.0, 20.0
        self.assertAlmostEqual(mf.dec(a - 0.0001, a, b), 1)
        self.assertAlmostEqual(mf.dec(b + 0.0001, a, b), 0)

    def test_produces_continuous_monotonic_decreasing_output(self):
        a, b = 10, 20
        x_values = np.linspace(5, 25, 100)
        y_values = [mf.dec(x, a, b) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertGreaterEqual(y_values[i], y_values[i + 1])

    def test_handles_single_point_interval_a_equals_b(self):
        self.assertEqual(mf.dec(9.99, 10, 10), 1)
        self.assertEqual(mf.dec(10.00, 10, 10), 1)
        self.assertEqual(mf.dec(10.01, 10, 10), 0)

    def test_returns_exactly_half_at_midpoint(self):
        a, b = 0, 10
        midpoint = (a + b) / 2
        self.assertAlmostEqual(mf.dec(midpoint, a, b), 0.5)

    def test_handles_co2_sensor_range(self):
        a, b = 400, 1500
        self.assertEqual(mf.dec(350, a, b), 1)
        self.assertEqual(mf.dec(1500, a, b), 0)
        self.assertAlmostEqual(mf.dec(950, a, b), (1500 - 950) / (1500 - 400))

    def test_handles_temperature_sensor_range(self):
        a, b = 18, 30
        self.assertEqual(mf.dec(10, a, b), 1)
        self.assertEqual(mf.dec(30, a, b), 0)
        self.assertAlmostEqual(mf.dec(24, a, b), 0.5)

    def test_handles_humidity_sensor_range(self):
        a, b = 30, 80
        self.assertEqual(mf.dec(25, a, b), 1)
        self.assertEqual(mf.dec(80, a, b), 0)
        self.assertAlmostEqual(mf.dec(55, a, b), 0.5)

    def test_returns_scalar_for_scalar_input(self):
        result = mf.dec(15, 10, 20)
        self.assertIsInstance(result, (int, float))

    def test_maintains_mathematical_linearity_in_middle_region(self):
        a, b = 10, 20
        x1, x2 = 12, 14
        y1 = mf.dec(x1, a, b)
        y2 = mf.dec(x2, a, b)

        slope = (y2 - y1) / (x2 - x1)
        expected_slope = -1 / (b - a)
        self.assertAlmostEqual(slope, expected_slope)

    def test_handles_inverted_range_gracefully(self):
        result = mf.dec(15, 20, 10)
        self.assertIsNotNone(result)

    def test_handles_zero_crossing_interval(self):
        a, b = -5, 5
        self.assertEqual(mf.dec(-10, a, b), 1)
        self.assertEqual(mf.dec(5, a, b), 0)
        self.assertAlmostEqual(mf.dec(0, a, b), 0.5)

    def test_boundary_value_a_returns_one(self):
        for a in [0, 100, -50, 3.14]:
            self.assertEqual(mf.dec(a, a, a + 10), 1)

    def test_boundary_value_b_returns_zero(self):
        for b in [10, 100, -50, 3.14]:
            self.assertEqual(mf.dec(b, b - 10, b), 0)

    def test_handles_consecutive_calls_with_same_parameters(self):
        result1 = mf.dec(15, 10, 20)
        result2 = mf.dec(15, 10, 20)
        self.assertEqual(result1, result2)

    def test_output_is_complement_of_increasing_function(self):
        a, b = 10, 20
        test_values = [5, 10, 15, 20, 25]
        for x in test_values:
            self.assertAlmostEqual(mf.dec(x, a, b), 1 - mf.inc(x, a, b))

    def test_correctly_implements_falling_ramp_semantics(self):
        a, b = 0, 100
        self.assertEqual(mf.dec(a - 1, a, b), 1)
        self.assertGreater(mf.dec(a, a, b), 0.99)
        self.assertLess(mf.dec(b, a, b), 0.01)
        self.assertEqual(mf.dec(b + 1, a, b), 0)


if __name__ == "__main__":
    unittest.main()
