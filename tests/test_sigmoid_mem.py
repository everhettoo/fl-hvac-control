import unittest
import numpy as np
from mylibs.sigmoid_mem import sigmoid


class TestSigmoidMembershipFunction(unittest.TestCase):

    def test_returns_zero_point_five_at_midpoint_with_positive_steepness(self):
        self.assertAlmostEqual(sigmoid(0, 1, 0), 0.5)
        self.assertAlmostEqual(sigmoid(10, 1, 10), 0.5)
        self.assertAlmostEqual(sigmoid(-5, 1, -5), 0.5)

    def test_returns_zero_point_five_at_midpoint_with_negative_steepness(self):
        self.assertAlmostEqual(sigmoid(0, -1, 0), 0.5)
        self.assertAlmostEqual(sigmoid(10, -1, 10), 0.5)
        self.assertAlmostEqual(sigmoid(-5, -1, -5), 0.5)

    def test_returns_values_between_zero_and_one_with_positive_steepness(self):
        result = sigmoid(5, 1, 0)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)
        self.assertGreater(result, 0.5)

    def test_returns_values_between_zero_and_one_with_negative_steepness(self):
        result = sigmoid(5, -1, 0)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)
        self.assertLess(result, 0.5)

    def test_increases_with_increasing_x_when_steepness_positive(self):
        c = 1
        x_values = [0, 1, 2, 3, 4]
        y_values = [sigmoid(x, c, 2) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertLess(y_values[i], y_values[i + 1])

    def test_decreases_with_increasing_x_when_steepness_negative(self):
        c = -1
        x_values = [0, 1, 2, 3, 4]
        y_values = [sigmoid(x, c, 2) for x in x_values]

        for i in range(len(y_values) - 1):
            self.assertGreater(y_values[i], y_values[i + 1])

    def test_steep_positive_slope_creates_sharp_transition(self):
        a = 10
        b = 0
        steep = sigmoid(b + 0.1, a, b)
        gentle = sigmoid(b + 0.1, 0.1, b)

        self.assertGreater(steep, gentle)

    def test_gentle_positive_slope_creates_smooth_transition(self):
        a = 0.1
        b = 0
        result = sigmoid(b + 1, a, b)
        self.assertGreater(result, 0.4)
        self.assertLess(result, 0.6)

    def test_steep_negative_slope_creates_sharp_transition(self):
        a = -10
        b = 0
        result_left = sigmoid(-1, a, b)
        result_center = sigmoid(0, a, b)
        result_right = sigmoid(1, a, b)

        self.assertGreater(result_left, 0.5)
        self.assertAlmostEqual(result_center, 0.5)
        self.assertLess(result_right, 0.5)

    def test_handles_negative_x_values(self):
        result = sigmoid(-5, 1, 0)
        self.assertGreater(result, 0)
        self.assertLess(result, 0.5)

    def test_handles_negative_midpoint(self):
        result = sigmoid(-5, 1, -5)
        self.assertAlmostEqual(result, 0.5)

    def test_handles_decimal_values(self):
        result = sigmoid(2.5, 0.5, 2.5)
        self.assertAlmostEqual(result, 0.5)

    def test_handles_very_large_steepness(self):
        a = 100
        b = 0
        result_left = sigmoid(-1, a, b)
        result_center = sigmoid(0, a, b)
        result_right = sigmoid(1, a, b)

        self.assertLess(result_left, 0.5)
        self.assertAlmostEqual(result_center, 0.5)
        self.assertGreater(result_right, 0.5)

    def test_handles_very_small_steepness(self):
        a = 0.001
        b = 0
        result_far_left = sigmoid(-100, a, b)
        result_at_center = sigmoid(0, a, b)
        result_far_right = sigmoid(100, a, b)

        self.assertGreater(result_far_left, 0.3)
        self.assertAlmostEqual(result_at_center, 0.5)
        self.assertLess(result_far_right, 0.7)

    def test_handles_x_as_numpy_array_with_positive_steepness(self):
        x_array = np.array([-5, -2.5, 0, 2.5, 5])
        result = sigmoid(x_array, 1, 0)

        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[2], 0.5)
        self.assertGreater(result[4], result[0])

    def test_handles_x_as_numpy_array_with_negative_steepness(self):
        x_array = np.array([-5, -2.5, 0, 2.5, 5])
        result = sigmoid(x_array, -1, 0)

        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[2], 0.5)
        self.assertLess(result[4], result[0])

    def test_returns_scalar_for_scalar_input(self):
        result = sigmoid(5, 1, 0)
        self.assertIsInstance(result, (float, np.floating))

    def test_returns_array_for_array_input(self):
        result = sigmoid(np.array([0, 1, 2]), 1, 0)
        self.assertIsInstance(result, np.ndarray)

    def test_produces_valid_membership_values_across_range(self):
        a = 1
        b = 0
        x_values = np.linspace(-10, 10, 100)
        results = sigmoid(x_values, a, b)

        for result in results:
            self.assertGreater(result, 0)
            self.assertLess(result, 1)

    def test_handles_consecutive_calls_with_same_parameters(self):
        result1 = sigmoid(5, 1, 0)
        result2 = sigmoid(5, 1, 0)
        self.assertEqual(result1, result2)

    def test_sigmoid_is_smooth_and_continuous_with_positive_steepness(self):
        a = 1
        b = 0
        x_values = np.linspace(-5, 5, 200)
        y_values = sigmoid(x_values, a, b)

        for i in range(len(y_values) - 1):
            diff = abs(y_values[i] - y_values[i + 1])
            self.assertLess(diff, 0.02)

    def test_sigmoid_is_smooth_and_continuous_with_negative_steepness(self):
        a = -1
        b = 0
        x_values = np.linspace(-5, 5, 200)
        y_values = sigmoid(x_values, a, b)

        for i in range(len(y_values) - 1):
            diff = abs(y_values[i] - y_values[i + 1])
            self.assertLess(diff, 0.02)

    def test_approaches_one_far_right_with_positive_steepness(self):
        a = 1
        b = 0
        result = sigmoid(100, a, b)
        self.assertGreater(result, 0.9999)
        self.assertLessEqual(result, 1)

    def test_approaches_zero_far_left_with_positive_steepness(self):
        a = 1
        b = 0
        result = sigmoid(-100, a, b)
        self.assertLess(result, 0.00001)
        self.assertGreater(result, 0)

    def test_approaches_zero_far_right_with_negative_steepness(self):
        a = -1
        b = 0
        result = sigmoid(100, a, b)
        self.assertLess(result, 0.00001)
        self.assertGreater(result, 0)

    def test_approaches_one_far_left_with_negative_steepness(self):
        a = -1
        b = 0
        result = sigmoid(-100, a, b)
        self.assertGreater(result, 0.9999)
        self.assertLessEqual(result, 1)

    def test_handles_co2_sensor_range_with_positive_steepness(self):
        midpoint = 800
        steepness = 0.01

        self.assertAlmostEqual(sigmoid(midpoint, steepness, midpoint), 0.5)
        self.assertGreater(sigmoid(1000, steepness, midpoint), 0.5)
        self.assertLess(sigmoid(600, steepness, midpoint), 0.5)

    def test_handles_temperature_sensor_range_with_positive_steepness(self):
        midpoint = 23
        steepness = 1

        self.assertAlmostEqual(sigmoid(midpoint, steepness, midpoint), 0.5)
        self.assertGreater(sigmoid(25, steepness, midpoint), 0.5)
        self.assertLess(sigmoid(21, steepness, midpoint), 0.5)

    def test_handles_humidity_sensor_range_with_positive_steepness(self):
        midpoint = 55
        steepness = 0.1

        self.assertAlmostEqual(sigmoid(midpoint, steepness, midpoint), 0.5)
        self.assertGreater(sigmoid(65, steepness, midpoint), 0.5)
        self.assertLess(sigmoid(45, steepness, midpoint), 0.5)

    def test_zero_crossing_interval_with_positive_steepness(self):
        a = 1
        b = 0
        self.assertLess(sigmoid(-10, a, b), 0.1)
        self.assertAlmostEqual(sigmoid(0, a, b), 0.5)
        self.assertGreater(sigmoid(10, a, b), 0.9)

    def test_zero_crossing_interval_with_negative_steepness(self):
        a = -1
        b = 0
        self.assertGreater(sigmoid(-10, a, b), 0.9)
        self.assertAlmostEqual(sigmoid(0, a, b), 0.5)
        self.assertLess(sigmoid(10, a, b), 0.1)

    def test_increasing_steepness_increases_slope_magnitude(self):
        b = 0
        x_test = 1
        result_mild = sigmoid(x_test, 0.5, b)
        result_steep = sigmoid(x_test, 2, b)

        mid_value = 0.5
        dist_mild = abs(result_mild - mid_value)
        dist_steep = abs(result_steep - mid_value)

        self.assertGreater(dist_steep, dist_mild)

    def test_multivariate_input_produces_corresponding_output_shape(self):
        x_values = np.array([1, 2, 3, 4, 5])
        results = sigmoid(x_values, 1, 2)

        self.assertEqual(results.shape, x_values.shape)

    def test_satisfies_sigmoid_mathematical_property_at_two_points(self):
        a = 2
        b = 0
        x1 = -1
        x2 = 1

        y1 = sigmoid(x1, a, b)
        y2 = sigmoid(x2, a, b)

        self.assertAlmostEqual(y1 + y2, 1, places=5)

    def test_handles_midpoint_translation(self):
        steepness = 1
        self.assertAlmostEqual(sigmoid(5, steepness, 5), 0.5)
        self.assertAlmostEqual(sigmoid(10, steepness, 10), 0.5)
        self.assertAlmostEqual(sigmoid(-5, steepness, -5), 0.5)

    def test_monotonically_increasing_sequence_with_positive_steepness(self):
        a = 1
        b = 0
        x_values = np.array([-3, -2, -1, 0, 1, 2, 3])
        y_values = sigmoid(x_values, a, b)

        for i in range(len(y_values) - 1):
            self.assertLessEqual(y_values[i], y_values[i + 1])

    def test_monotonically_decreasing_sequence_with_negative_steepness(self):
        a = -1
        b = 0
        x_values = np.array([-3, -2, -1, 0, 1, 2, 3])
        y_values = sigmoid(x_values, a, b)

        for i in range(len(y_values) - 1):
            self.assertGreaterEqual(y_values[i], y_values[i + 1])


if __name__ == "__main__":
    unittest.main()

