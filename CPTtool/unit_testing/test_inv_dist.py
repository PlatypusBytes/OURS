from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')
import inv_dist


class TestInverDist(TestCase):
    def setUp(self):
        np.random.seed(1)
        return

    def test_data_1(self):

        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.array(4)
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data)
        # predict
        interp.predict(testing.reshape(1, 1))

        # testing
        np.testing.assert_array_almost_equal(interp.zn, data[4])
        return

    def test_data_2(self):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.array(6)
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data)
        # predict
        interp.predict(testing.reshape(1, 1))

        # testing
        np.testing.assert_array_almost_equal(interp.zn, data[6])
        return

    def test_data_3(self):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.random.rand(10) * 10
        pw = 1
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=pw)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data)

        # for each testing point
        for i, t in enumerate(testing):
            # predict
            interp.predict(t.reshape(1, 1))
            # result
            dist = np.abs((position - t) + 1e-9)
            val = np.sum(data / (dist ** pw)) / np.sum(1 / (dist ** pw))

            # testing
            np.testing.assert_array_almost_equal(interp.zn, val)
        return

    def test_data_4(self):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.random.rand(10) * 10
        pw = 2
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=pw)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data)

        # for each testing point
        for i, t in enumerate(testing):
            # predict
            interp.predict(t.reshape(1, 1))
            # result
            dist = np.abs((position - t) + 1e-9)
            val = np.sum(data / (dist ** pw)) / np.sum(1 /(dist ** pw))

            # testing
            np.testing.assert_array_almost_equal(interp.zn, val)
        return

    def test_data_5(self):

        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.array(4)
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=0)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data)
        # predict
        interp.predict(testing.reshape(1, 1))

        # testing
        np.testing.assert_array_almost_equal(interp.zn, np.mean(data))
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

