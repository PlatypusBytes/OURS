from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')


class TestUtils(TestCase):
    def setUp(self):
        return

    def test_log_normal_1(self):
        r"""
        Tests the log normal distribution
        """
        import tools_utils as tu

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
        import tools_utils as tu

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

    def test_compute_probability_1(self):
        r"""
        Tests the compute_probability for the scenarios
        """
        import tools_utils as tu

        coord_scr = [5]
        coord_cpt = np.array([[0, 10]])
        coord_rec = [30]

        # compute probability
        probs = tu.compute_probability(coord_cpt, coord_scr, coord_rec)

        # test
        np.testing.assert_almost_equal(probs, [100])
        return

    def test_compute_probability_2(self):
        r"""
        Tests the compute_probability for the scenarios
        """
        import tools_utils as tu

        coord_scr = [5, 10]
        coord_cpt = np.array([[0, 10],
                     [17, 19],
                     [14, 22],
                     [35, 10]
                     ])
        coord_rec = [30, 10]

        # compute probability
        probs = tu.compute_probability(coord_cpt, coord_scr, coord_rec)

        # test
        np.testing.assert_almost_equal(probs, [30.696406171699,
                                               26.369271616348,
                                               25.422551848429,
                                               17.511770363525])
        return

    def test_ceil_value_1(self):
        r"""
        Tests the data replacement
        """
        import tools_utils as tu

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
        import tools_utils as tu

        data = [0, 0, 20, 0, 40]

        # compute probability
        new_data = tu.ceil_value(data, 0)

        # test
        np.testing.assert_almost_equal(new_data, [20, 20, 20, 40, 40])
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

