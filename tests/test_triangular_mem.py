import unittest
import numpy as np
import mylibs.membership_functions as mf

class TestTriangularMembershipFunction(unittest.TestCase):

    def test_returns_zero_when_x_less_than_a(self):
        self.assertEqual(mf.tri(5, 10, 15, 20), 0)
        self.assertEqual(mf.tri(-100, 0, 50, 100), 0)

    def test_returns_zero_when_x_equal_to_a(self):
        self.assertEqual(mf.tri(10, 10, 15, 20), 0)
        self.assertEqual(mf.tri(0, 0, 50, 100), 0)

    def test_returns_zero_when_x_greater_than_c(self):
        self.assertEqual(mf.tri(25, 10, 15, 20), 0)
        self.assertEqual(mf.tri(150, 0, 50, 100), 0)

    def test_returns_zero_when_x_equal_to_c(self):
        self.assertEqual(mf.tri(20, 10, 15, 20), 0)
        self.assertEqual(mf.tri(100, 0, 50, 100), 0)

    def test_returns_one_at_peak_b(self):
        self.assertEqual(mf.tri(15, 10, 15, 20), 1)
        self.assertEqual(mf.tri(50, 0, 50, 100), 1)
        self.assertEqual(mf.tri(0, -10, 0, 10), 1)

    def test_returns_half_at_midpoint_between_a_and_b(self):
        self.assertAlmostEqual(mf.tri(12.5, 10, 15, 20), 0.5)
        self.assertAlmostEqual(mf.tri(25, 0, 50, 100), 0.5)

    def test_returns_half_at_midpoint_between_b_and_c(self):
        self.assertAlmostEqual(mf.tri(17.5, 10, 15, 20), 0.5)
        self.assertAlmostEqual(mf.tri(75, 0, 50, 100), 0.5)

    def test_handles_left_slope_linear_interpolation(self):
        self.assertAlmostEqual(mf.tri(11, 10, 15, 20), 0.2)
        self.assertAlmostEqual(mf.tri(13, 10, 15, 20), 0.6)

    def test_handles_right_slope_linear_interpolation(self):
        self.assertAlmostEqual(mf.tri(19, 10, 15, 20), 0.2)
        self.assertAlmostEqual(mf.tri(17, 10, 15, 20), 0.6)

    def test_handles_negative_ranges(self):
        self.assertEqual(mf.tri(-25, -20, -10, 0), 0)
        self.assertAlmostEqual(mf.tri(-10, -20, -10, 0), 1)
        self.assertAlmostEqual(mf.tri(-15, -20, -10, 0), 0.5)
        self.assertEqual(mf.tri(0, -20, -10, 0), 0)

    def test_handles_decimal_values(self):
        self.assertAlmostEqual(mf.tri(2.5, 1.0, 2.5, 4.0), 1)
        self.assertAlmostEqual(mf.tri(1.75, 1.0, 2.5, 4.0), 0.5)
        self.assertAlmostEqual(mf.tri(3.25, 1.0, 2.5, 4.0), 0.5)

    def test_handles_very_small_interval(self):
        result = mf.tri(1.0001, 1.0, 1.00015, 1.0002)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_handles_very_large_values(self):
        self.assertLess(mf.tri(999999, 500000, 750000, 1000000), 0.001)
        self.assertEqual(mf.tri(750000, 500000, 750000, 1000000), 1)
        self.assertAlmostEqual(mf.tri(625000, 500000, 750000, 1000000), 0.5)

    def test_handles_a_equals_b_case(self):
        self.assertEqual(mf.tri(10, 10, 10, 20), 0)
        self.assertGreater(mf.tri(10.0001, 10, 10, 20), 0.99)
        self.assertEqual(mf.tri(20, 10, 10, 20), 0)
        self.assertAlmostEqual(mf.tri(15, 10, 10, 20), 0.5)

    def test_handles_b_equals_c_case(self):
        self.assertEqual(mf.tri(10, 10, 20, 20), 0)
        self.assertEqual(mf.tri(20, 10, 20, 20), 1)
        self.assertAlmostEqual(mf.tri(15, 10, 20, 20), 0.5)

    def test_handles_a_equals_b_equals_c_case(self):
        self.assertAlmostEqual(mf.tri(10, 10, 10, 10), 0)
        self.assertEqual(mf.tri(10.0001, 10, 10, 10), 0)
        self.assertEqual(mf.tri(9.9999, 10, 10, 10), 0)

    def test_handles_x_as_numpy_array(self):
        x_array = np.array([5, 10, 12.5, 15, 17.5, 20, 25])
        result = np.array([mf.tri(val, 10, 15, 20) for val in x_array])
        expected = np.array([0, 0, 0.5, 1, 0.5, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_float_precision_at_boundaries(self):
        self.assertAlmostEqual(mf.tri(10 - 0.0001, 10, 15, 20), 0)
        self.assertAlmostEqual(mf.tri(20 + 0.0001, 10, 15, 20), 0)
        self.assertGreater(mf.tri(10.0001, 10, 15, 20), 0)
        self.assertLess(mf.tri(19.9999, 10, 15, 20), 0.001)

    def test_produces_continuous_output_across_interval(self):
        a, b, c = 10, 15, 20
        x_values = np.linspace(8, 22, 200)
        y_values = [mf.tri(x, a, b, c) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertLessEqual(abs(y_values[i] - y_values[i + 1]), 0.02)

    def test_increases_from_zero_to_one_on_left_slope(self):
        a, b, c = 10, 15, 20
        x_values = np.linspace(a + 0.001, b - 0.001, 50)
        y_values = [mf.tri(x, a, b, c) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertLessEqual(y_values[i], y_values[i + 1])

    def test_decreases_from_one_to_zero_on_right_slope(self):
        a, b, c = 10, 15, 20
        x_values = np.linspace(b + 0.001, c - 0.001, 50)
        y_values = [mf.tri(x, a, b, c) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertGreaterEqual(y_values[i], y_values[i + 1])

    def test_returns_scalar_for_scalar_input(self):
        result = mf.tri(15, 10, 15, 20)
        self.assertIsInstance(result, (int, float))

    def test_maintains_mathematical_linearity_on_left_slope(self):
        a, b, c = 10, 15, 20
        x1, x2 = 11, 13
        y1 = mf.tri(x1, a, b, c)
        y2 = mf.tri(x2, a, b, c)

        slope = (y2 - y1) / (x2 - x1)
        expected_slope = 1 / (b - a)
        self.assertAlmostEqual(slope, expected_slope)

    def test_maintains_mathematical_linearity_on_right_slope(self):
        a, b, c = 10, 15, 20
        x1, x2 = 16, 18
        y1 = mf.tri(x1, a, b, c)
        y2 = mf.tri(x2, a, b, c)

        slope = (y2 - y1) / (x2 - x1)
        expected_slope = -1 / (c - b)
        self.assertAlmostEqual(slope, expected_slope)

    def test_handles_zero_crossing_interval(self):
        a, b, c = -5, 0, 5
        self.assertEqual(mf.tri(-10, a, b, c), 0)
        self.assertEqual(mf.tri(0, a, b, c), 1)
        self.assertAlmostEqual(mf.tri(-2.5, a, b, c), 0.5)
        self.assertEqual(mf.tri(10, a, b, c), 0)

    def test_handles_co2_sensor_range(self):
        a, b, c = 600, 700, 1000
        self.assertEqual(mf.tri(550, a, b, c), 0)
        self.assertEqual(mf.tri(700, a, b, c), 1)
        self.assertAlmostEqual(mf.tri(650, a, b, c), 0.5)
        self.assertEqual(mf.tri(1050, a, b, c), 0)

    def test_handles_temperature_sensor_range(self):
        a, b, c = 20, 23.5, 27
        self.assertEqual(mf.tri(19, a, b, c), 0)
        self.assertEqual(mf.tri(23.5, a, b, c), 1)
        self.assertAlmostEqual(mf.tri(21.75, a, b, c), 0.5)
        self.assertEqual(mf.tri(28, a, b, c), 0)

    def test_handles_humidity_sensor_range(self):
        a, b, c = 40, 52.5, 65
        self.assertEqual(mf.tri(35, a, b, c), 0)
        self.assertEqual(mf.tri(52.5, a, b, c), 1)
        self.assertAlmostEqual(mf.tri(46.25, a, b, c), 0.5)
        self.assertEqual(mf.tri(70, a, b, c), 0)

    def test_handles_consecutive_calls_with_same_parameters(self):
        result1 = mf.tri(15, 10, 15, 20)
        result2 = mf.tri(15, 10, 15, 20)
        self.assertEqual(result1, result2)

    def test_symmetric_triangle_has_equal_slopes(self):
        a, b, c = 10, 15, 20
        left_slope = 1 / (b - a)
        right_slope = 1 / (c - b)
        self.assertAlmostEqual(left_slope, right_slope)

    def test_produces_valid_membership_values_in_full_range(self):
        a, b, c = 10, 15, 20
        test_points = np.linspace(5, 25, 100)

        for x in test_points:
            result = mf.tri(x, a, b, c)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)

    def test_peak_is_unique_maximum(self):
        a, b, c = 10, 15, 20
        left_val = mf.tri(14.9, a, b, c)
        peak_val = mf.tri(15, a, b, c)
        right_val = mf.tri(15.1, a, b, c)

        self.assertLess(left_val, peak_val)
        self.assertLess(right_val, peak_val)
        self.assertEqual(peak_val, 1)


if __name__ == "__main__":
    unittest.main()

