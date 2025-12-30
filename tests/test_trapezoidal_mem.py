import unittest
import numpy as np
from mylibs.trapeziod_mem import trap


class TestTrapezoidalMembershipFunction(unittest.TestCase):

    def test_returns_zero_when_x_less_than_a(self):
        self.assertEqual(trap(5, 10, 15, 20, 25), 0)
        self.assertEqual(trap(-100, 0, 25, 75, 100), 0)

    def test_returns_zero_when_x_equal_to_a(self):
        self.assertEqual(trap(10, 10, 15, 20, 25), 0)
        self.assertEqual(trap(0, 0, 25, 75, 100), 0)

    def test_returns_zero_when_x_greater_than_d(self):
        self.assertEqual(trap(30, 10, 15, 20, 25), 0)
        self.assertEqual(trap(150, 0, 25, 75, 100), 0)

    def test_returns_zero_when_x_equal_to_d(self):
        self.assertEqual(trap(25, 10, 15, 20, 25), 0)
        self.assertEqual(trap(100, 0, 25, 75, 100), 0)

    def test_returns_one_on_plateau_at_b(self):
        self.assertEqual(trap(15, 10, 15, 20, 25), 1)
        self.assertEqual(trap(25, 0, 25, 75, 100), 1)

    def test_returns_one_on_plateau_at_c(self):
        self.assertEqual(trap(20, 10, 15, 20, 25), 1)
        self.assertEqual(trap(75, 0, 25, 75, 100), 1)

    def test_returns_one_on_plateau_between_b_and_c(self):
        self.assertAlmostEqual(trap(17.5, 10, 15, 20, 25), 1)
        self.assertAlmostEqual(trap(50, 0, 25, 75, 100), 1)

    def test_handles_left_slope_linear_interpolation(self):
        self.assertAlmostEqual(trap(12.5, 10, 15, 20, 25), 0.5)
        self.assertAlmostEqual(trap(11, 10, 15, 20, 25), 0.2)
        self.assertAlmostEqual(trap(13, 10, 15, 20, 25), 0.6)

    def test_handles_right_slope_linear_interpolation(self):
        self.assertAlmostEqual(trap(22.5, 10, 15, 20, 25), 0.5)
        self.assertAlmostEqual(trap(24, 10, 15, 20, 25), 0.2)
        self.assertAlmostEqual(trap(23, 10, 15, 20, 25), 0.4)

    def test_handles_negative_ranges(self):
        self.assertEqual(trap(-30, -20, -15, -10, 0), 0)
        self.assertAlmostEqual(trap(-17.5, -20, -15, -10, 0), 0.5)
        self.assertEqual(trap(0, -20, -15, -10, 0), 0)

    def test_handles_decimal_values(self):
        self.assertAlmostEqual(trap(7.5, 5.0, 7.5, 12.5, 15.0), 1)
        self.assertAlmostEqual(trap(6.25, 5.0, 7.5, 12.5, 15.0), 0.5)
        self.assertAlmostEqual(trap(13.75, 5.0, 7.5, 12.5, 15.0), 0.5)

    def test_handles_very_small_interval(self):
        result = trap(1.0001, 1.0, 1.00025, 1.00035, 1.0005)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)

    def test_handles_very_large_values(self):
        self.assertEqual(trap(499999, 500000, 625000, 875000, 1000000), 0)
        self.assertAlmostEqual(trap(750000, 500000, 625000, 875000, 1000000), 1)
        self.assertLess(trap(999999, 500000, 625000, 875000, 1000000), 0.001)

    def test_handles_a_equals_b_case(self):
        self.assertGreater(trap(10.0001, 10, 10, 20, 25), 0.99)
        self.assertEqual(trap(20, 10, 10, 20, 25), 1)
        self.assertEqual(trap(25, 10, 10, 20, 25), 0)

    def test_handles_c_equals_d_case(self):
        self.assertEqual(trap(10, 10, 15, 25, 25), 0)
        self.assertEqual(trap(20, 10, 15, 25, 25), 1)
        self.assertGreater(trap(24.9999, 10, 15, 25, 25), 0.99)

    def test_handles_b_equals_c_case(self):
        self.assertEqual(trap(10, 10, 20, 20, 25), 0)
        self.assertEqual(trap(20, 10, 20, 20, 25), 1)
        self.assertAlmostEqual(trap(15, 10, 20, 20, 25), 0.5)

    def test_handles_a_equals_b_equals_c_case(self):
        self.assertGreater(trap(10.0001, 10, 10, 10, 25), 0.99)
        self.assertEqual(trap(25, 10, 10, 10, 25), 0)
        self.assertAlmostEqual(trap(17.5, 10, 10, 10, 25), 0.5)

    def test_handles_b_equals_c_equals_d_case(self):
        self.assertEqual(trap(10, 10, 20, 20, 20), 0)
        self.assertEqual(trap(20, 10, 20, 20, 20), 1)
        self.assertAlmostEqual(trap(15, 10, 20, 20, 20), 0.5)

    def test_handles_a_equals_b_equals_c_equals_d_case(self):
        self.assertEqual(trap(10, 10, 10, 10, 10), 0)
        self.assertAlmostEqual(trap(10.0001, 10, 10, 10, 10), 0)

    def test_handles_x_as_numpy_array(self):
        x_array = np.array([5, 10, 12.5, 15, 20, 22.5, 25, 30])
        result = np.array([trap(val, 10, 15, 20, 25) for val in x_array])
        expected = np.array([0, 0, 0.5, 1, 1, 0.5, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_float_precision_at_boundaries(self):
        self.assertAlmostEqual(trap(10 - 0.0001, 10, 15, 20, 25), 0)
        self.assertAlmostEqual(trap(25 + 0.0001, 10, 15, 20, 25), 0)
        self.assertGreater(trap(10.0001, 10, 15, 20, 25), 0)
        self.assertLess(trap(24.9999, 10, 15, 20, 25), 0.001)

    def test_produces_continuous_output_across_interval(self):
        a, b, c, d = 10, 15, 20, 25
        x_values = np.linspace(8, 27, 200)
        y_values = [trap(x, a, b, c, d) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertLessEqual(abs(y_values[i] - y_values[i + 1]), 0.02)

    def test_increases_from_zero_to_one_on_left_slope(self):
        a, b, c, d = 10, 15, 20, 25
        x_values = np.linspace(a + 0.001, b - 0.001, 50)
        y_values = [trap(x, a, b, c, d) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertLessEqual(y_values[i], y_values[i + 1])

    def test_maintains_plateau_at_one_between_b_and_c(self):
        a, b, c, d = 10, 15, 20, 25
        x_values = np.linspace(b, c, 20)
        y_values = [trap(x, a, b, c, d) for x in x_values]

        for y in y_values:
            self.assertAlmostEqual(y, 1)

    def test_decreases_from_one_to_zero_on_right_slope(self):
        a, b, c, d = 10, 15, 20, 25
        x_values = np.linspace(c + 0.001, d - 0.001, 50)
        y_values = [trap(x, a, b, c, d) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertGreaterEqual(y_values[i], y_values[i + 1])

    def test_returns_scalar_for_scalar_input(self):
        result = trap(17.5, 10, 15, 20, 25)
        self.assertIsInstance(result, (int, float))

    def test_maintains_mathematical_linearity_on_left_slope(self):
        a, b, c, d = 10, 15, 20, 25
        x1, x2 = 11, 13
        y1 = trap(x1, a, b, c, d)
        y2 = trap(x2, a, b, c, d)

        slope = (y2 - y1) / (x2 - x1)
        expected_slope = 1 / (b - a)
        self.assertAlmostEqual(slope, expected_slope)

    def test_maintains_mathematical_linearity_on_right_slope(self):
        a, b, c, d = 10, 15, 20, 25
        x1, x2 = 21, 23
        y1 = trap(x1, a, b, c, d)
        y2 = trap(x2, a, b, c, d)

        slope = (y2 - y1) / (x2 - x1)
        expected_slope = -1 / (d - c)
        self.assertAlmostEqual(slope, expected_slope)

    def test_handles_zero_crossing_interval(self):
        a, b, c, d = -10, -5, 5, 10
        self.assertEqual(trap(-15, a, b, c, d), 0)
        self.assertAlmostEqual(trap(-7.5, a, b, c, d), 0.5)
        self.assertEqual(trap(0, a, b, c, d), 1)
        self.assertAlmostEqual(trap(7.5, a, b, c, d), 0.5)
        self.assertEqual(trap(15, a, b, c, d), 0)

    def test_handles_co2_sensor_range(self):
        a, b, c, d = 550, 600, 900, 1050
        self.assertEqual(trap(500, a, b, c, d), 0)
        self.assertAlmostEqual(trap(575, a, b, c, d), 0.5)
        self.assertEqual(trap(700, a, b, c, d), 1)
        self.assertAlmostEqual(trap(975, a, b, c, d), 0.5)
        self.assertEqual(trap(1100, a, b, c, d), 0)

    def test_handles_temperature_sensor_range(self):
        a, b, c, d = 20, 22, 25, 27
        self.assertEqual(trap(19, a, b, c, d), 0)
        self.assertAlmostEqual(trap(21, a, b, c, d), 0.5)
        self.assertEqual(trap(23.5, a, b, c, d), 1)
        self.assertAlmostEqual(trap(26, a, b, c, d), 0.5)
        self.assertEqual(trap(28, a, b, c, d), 0)

    def test_handles_humidity_sensor_range(self):
        a, b, c, d = 40, 45, 60, 65
        self.assertEqual(trap(35, a, b, c, d), 0)
        self.assertAlmostEqual(trap(42.5, a, b, c, d), 0.5)
        self.assertEqual(trap(50, a, b, c, d), 1)
        self.assertAlmostEqual(trap(62.5, a, b, c, d), 0.5)
        self.assertEqual(trap(70, a, b, c, d), 0)

    def test_handles_consecutive_calls_with_same_parameters(self):
        result1 = trap(17.5, 10, 15, 20, 25)
        result2 = trap(17.5, 10, 15, 20, 25)
        self.assertEqual(result1, result2)

    def test_produces_valid_membership_values_in_full_range(self):
        a, b, c, d = 10, 15, 20, 25
        test_points = np.linspace(5, 30, 100)

        for x in test_points:
            result = trap(x, a, b, c, d)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)

    def test_trapezoid_with_wide_plateau(self):
        a, b, c, d = 0, 10, 90, 100
        self.assertEqual(trap(50, a, b, c, d), 1)
        self.assertEqual(trap(90, a, b, c, d), 1)
        self.assertAlmostEqual(trap(5, a, b, c, d), 0.5)
        self.assertAlmostEqual(trap(95, a, b, c, d), 0.5)

    def test_trapezoid_with_narrow_plateau(self):
        a, b, c, d = 10, 15, 15.1, 20
        self.assertEqual(trap(15, a, b, c, d), 1)
        self.assertAlmostEqual(trap(12.5, a, b, c, d), 0.5)
        self.assertGreater(trap(17.5, a, b, c, d), 0.4)
        self.assertLess(trap(17.5, a, b, c, d), 0.6)

    def test_asymmetric_trapezoid_steep_left_gentle_right(self):
        a, b, c, d = 10, 11, 20, 40
        self.assertAlmostEqual(trap(10.5, a, b, c, d), 0.5)
        self.assertAlmostEqual(trap(30, a, b, c, d), 0.5)
        left_slope = 1 / (11 - 10)
        right_slope = 1 / (40 - 20)
        self.assertGreater(left_slope, right_slope)

    def test_asymmetric_trapezoid_gentle_left_steep_right(self):
        a, b, c, d = 0, 20, 30, 31
        self.assertAlmostEqual(trap(10, a, b, c, d), 0.5)
        self.assertAlmostEqual(trap(30.5, a, b, c, d), 0.5)
        left_slope = 1 / (20 - 0)
        right_slope = 1 / (31 - 30)
        self.assertLess(left_slope, right_slope)

    def test_symmetric_trapezoid_equal_slopes(self):
        a, b, c, d = 10, 15, 20, 25
        left_slope = 1 / (b - a)
        right_slope = 1 / (d - c)
        self.assertAlmostEqual(left_slope, right_slope)


if __name__ == "__main__":
    unittest.main()

