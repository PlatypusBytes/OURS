from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')
import tools_utils as tu


class TestUtils(TestCase):
    def setUp(self):
        return

    def test_log_normal_1(self):
        r"""
        Tests the log normal distribution
        """
        # define values
        x = np.ones(10)

        # compute mean and std
        mean, std = tu.log_normal_parameters(x)

        # test
        np.testing.assert_almost_equal(mean, 1)
        np.testing.assert_almost_equal(std, 0)
        return

    def test_log_normal_2(self):
        r"""
        Tests the log normal distribution
        Performs 10 tests in a for loop
        """

        # loop 10 times
        for i in range(10):
            # define values
            x = np.random.rand(10)

            # compute mean and std
            mean, std = tu.log_normal_parameters(x)

            # compute mean and standard deviation: analytical
            mu = np.mean(np.log(x))
            sigma = np.std(np.log(x))

            m = np.exp(mu + sigma**2 / 2)
            v = np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)

            # test
            np.testing.assert_almost_equal(mean, m)
            np.testing.assert_almost_equal(std, np.sqrt(v))
        return

    # def test_compute_probability_1(self):
    #     r"""
    #     Tests the compute_probability for the scenarios
    #     """
    #     import tools_utils as tu
    #
    #     coord_scr = [5]
    #     coord_cpt = np.array([[0, 10]])
    #     coord_rec = [30]
    #
    #     # compute probability
    #     probs = tu.compute_probability(coord_cpt, coord_scr, coord_rec)
    #
    #     # test
    #     np.testing.assert_almost_equal(probs, [100])
    #     return
    #
    # def test_compute_probability_2(self):
    #     r"""
    #     Tests the compute_probability for the scenarios
    #     """
    #     import tools_utils as tu
    #
    #     coord_scr = [5, 10]
    #     coord_cpt = np.array([[0, 10],
    #                  [17, 19],
    #                  [14, 22],
    #                  [35, 10]
    #                  ])
    #     coord_rec = [30, 10]
    #
    #     # compute probability
    #     probs = tu.compute_probability(coord_cpt, coord_scr, coord_rec)
    #
    #     # test
    #     np.testing.assert_almost_equal(probs, [30.696406171699,
    #                                            26.369271616348,
    #                                            25.422551848429,
    #                                            17.511770363525])
    #     return

    def test_ceil_value_1(self):
        r"""
        Tests the data replacement
        """

        data = [0, 0, 20, 30, 40]

        # compute probability
        new_data = tu.ceil_value(data, 0)

        # test
        np.testing.assert_almost_equal(new_data, [20, 20, 20, 30, 40])
        return

    def test_ceil_value_2(self):
        r"""
        Tests the data replacement
        """
        import tools_utils as tu

        data = [0, -10, 20, 30, 40]

        # compute probability
        new_data = tu.ceil_value(data, 0)

        # test
        np.testing.assert_almost_equal(new_data, [20, 20, 20, 30, 40])
        return

    def test_ceil_value_3(self):
        r"""
        Tests the data replacement
        """

        data = [0, 0, 20, 0, 40]

        # compute probability
        new_data = tu.ceil_value(data, 0)

        # test
        np.testing.assert_almost_equal(new_data, [20, 20, 20, 40, 40])
        return

    def test_merge_thickness_1(self):
        # Set all the values
        min_layer_thick = 0.5
        cpt = {}
        cpt.update({"depth": np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
                    "Qtn": np.array([2, 2, 2, 2, 2, 1.1, 2, 2, 2, 2, 2, 2]),
                    "Fr": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
                    })

        # Run the function to be tested
        depth_json, indx_json, lithology_json = tu.merge_thickness(cpt, min_layer_thick)

        # Set the target results
        depth_test = [0.0, 0.5, 1.3]
        test_lithology = ['1', r'2/3']
        test_index = [0, 5, 11]

        # Check if they are equal
        np.testing.assert_array_equal(depth_test, depth_json)
        np.testing.assert_array_equal(test_lithology, lithology_json)
        np.testing.assert_array_equal(test_index, indx_json)
        return

    def test_merge_thickness_2(self):
        # Set all the values
        min_layer_thick = 0.5
        cpt = {}
        cpt.update({"depth": np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
                    "Qtn": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000]),
                    "Fr": np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 2, 2])
                    })

        # Run the function to be tested
        depth_json, indx_json, lithology_json = tu.merge_thickness(cpt, min_layer_thick)

        # Set the target results
        depth_test = [0.0, 0.5, 1.3]
        test_lithology = ['1', r'2/8']
        test_index = [0, 5, 11]

        # Check if they are equal
        np.testing.assert_array_equal(depth_test, depth_json)
        np.testing.assert_array_equal(test_lithology, lithology_json)
        np.testing.assert_array_equal(test_index, indx_json)
        return

    def test_merge_thickness_3(self):
        # Set all the values
        min_layer_thick = 0.5
        cpt = {}
        cpt.update({"depth": np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
                                       1.6, 1.7, 1.8, 1.9, 2]),
                    "Qtn": np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 100, 100, 100, 100, 100, 100, 100]),
                    "Fr":  np.array([1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 2, 2, 2, 2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
                    })

        # Run the function to be tested
        depth_json, indx_json, lithology_json = tu.merge_thickness(cpt, min_layer_thick)

        # Results expected from the function
        depth_test = [0, 0.7, 1.4, 2.]
        test_lithology = ['1', r'2/3', '6']
        test_index = [0, 7, 14, 20]

        # Check if they are equal
        np.testing.assert_array_equal(depth_test, depth_json)
        np.testing.assert_array_equal(test_lithology, lithology_json)
        np.testing.assert_array_equal(test_index, indx_json)
        return

    def test_merge_thickness_4(self):

        # Set all the values
        min_layer_thick = 0.5
        cpt = {}
        cpt.update({"depth": np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]),
                    "Qtn": np.array([1, 1, 1, 1, 1, 1, 1, 10, 5, 1, 10, 5, 1, 10, 5, 1]),
                    "Fr":  np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
                    })

        # Run the function to be tested
        depth_json, indx_json, lithology_json = tu.merge_thickness(cpt, min_layer_thick)

        # Results expected from the function
        depth_test = [0, 0.7, 1.5]
        test_lithology = ['1', r'4/3/2']
        test_index = [0, 7, 15]

        # Check if they are equal
        np.testing.assert_array_equal(depth_test, depth_json)
        np.testing.assert_array_equal(test_lithology, lithology_json)
        np.testing.assert_array_equal(test_index, indx_json)
        return

    def tearDown(self):
        return


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity
    from teamcity.unittestpy import TeamcityTestRunner
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)

