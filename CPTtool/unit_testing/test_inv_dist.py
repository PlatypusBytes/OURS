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
        interp.interpolate(position.reshape((len(position), 1)), data, position, position)
        # predict
        interp.predict(testing.reshape(1, 1), point=True)

        # testing
        np.testing.assert_array_almost_equal(interp.zn[0], data[4])
        return

    def test_data_2(self):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.array(6)
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data, position, position)
        # predict
        interp.predict(testing.reshape(1, 1), point=True)

        # testing
        np.testing.assert_array_almost_equal(interp.zn[0], data[6])
        return

    def test_data_3(self):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.random.rand(10) * 10
        pw = 1
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=pw)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data, position, position)

        # for each testing point
        for i, t in enumerate(testing):
            # predict
            interp.predict(t.reshape(1, 1), point=True)
            # result
            dist = np.abs((position - t) + 1e-9)
            val = np.sum(data / (dist ** pw)) / np.sum(1 / (dist ** pw))

            # testing
            np.testing.assert_array_almost_equal(interp.zn[0], val)
        return

    def test_data_4(self):
        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.random.rand(10) * 10
        pw = 2
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=pw)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data, position, position)

        # for each testing point
        for i, t in enumerate(testing):
            # predict
            interp.predict(t.reshape(1, 1), point=True)
            # result
            dist = np.abs((position - t) + 1e-9)
            val = np.sum(data / (dist ** pw)) / np.sum(1 /(dist ** pw))

            # testing
            np.testing.assert_array_almost_equal(interp.zn[0], val)
        return

    def test_data_5(self):

        position = np.linspace(0, 10, 11)
        data = np.array(np.random.rand(len(position)))
        testing = np.array(4)
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=0)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 1)), data, position, position)
        # predict
        interp.predict(testing.reshape(1, 1), point=True)

        # testing
        np.testing.assert_array_almost_equal(interp.zn[0], np.mean(data))
        return

    def test_data_6(self):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10),
                np.ones(10) * 2]
        testing = np.array([0.5, 0.5])
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)], np.linspace(0, 10, 10))
        # predict
        interp.predict(testing.reshape(1, 2))

        # testing
        np.testing.assert_array_almost_equal(interp.zn, np.ones(10) * 1.5)
        return

    def test_data_7(self):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10),
                np.ones(10) * 2]
        testing = np.array([0, 0])
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)], np.linspace(0, 10, 10))
        # predict
        interp.predict(testing.reshape(1, 2))

        # testing
        np.testing.assert_array_almost_equal(interp.zn, np.ones(10) * 1.)
        return

    def test_data_8(self):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10),
                np.ones(10) * 2]
        testing = np.array([0.5, 0.5])
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)], np.linspace(0, 10, 10))
        # predict
        interp.predict(testing.reshape(1, 2))

        # analytical result
        w1 = 1 / np.linalg.norm(position[0]-testing) ** 1
        w2 = 1 / np.linalg.norm(position[1]-testing) ** 1
        solution = (1. * w1 + 2. * w2 * np.ones(10)) / (w1 + w2)
        # testing
        np.testing.assert_array_almost_equal(interp.zn, solution)
        return

    def test_data_9(self):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10),
                np.ones(10) * 2]
        testing = np.array([0.9, 0.9])
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)], np.linspace(0, 10, 10))
        # predict
        interp.predict(testing.reshape(1, 2))

        # analytical result
        w1 = 1 / np.linalg.norm(position[0] - testing) ** 1
        w2 = 1 / np.linalg.norm(position[1] - testing) ** 1
        solution = (1. * w1 + 2. * w2 * np.ones(10)) / (w1 + w2)
        # testing
        np.testing.assert_array_almost_equal(interp.zn, solution)
        return

    def test_data_10(self):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10),
                np.ones(10) * 2]
        testing = np.array([1.5, 1.5])
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data, [np.linspace(0, 10, 10), np.linspace(0, 10, 10)], np.linspace(0, 10, 10))
        # predict
        interp.predict(testing.reshape(1, 2))

        # analytical result
        w1 = 1 / np.linalg.norm(position[0] - testing) ** 1
        w2 = 1 / np.linalg.norm(position[1] - testing) ** 1
        solution = (1. * w1 + 2. * w2 * np.ones(10)) / (w1 + w2)
        # testing
        np.testing.assert_array_almost_equal(interp.zn, solution)
        return

    def test_data_11(self):

        position = np.array([[0, 0], [1, 1]])
        data = [np.ones(10),
                np.ones(10) * 2]
        testing = np.array([1.5, 1.5])
        # interpolate the top and bottom depth at this point
        interp = inv_dist.InverseDistance(nb_points=len(data), pwr=1)
        # create interpolation object
        interp.interpolate(position.reshape((len(position), 2)), data, [np.linspace(2, 8, 10), np.linspace(0, 10, 10)], np.linspace(0, 10, 10))
        # predict
        interp.predict(testing.reshape(1, 2))

        # analytical result
        w1 = 1 / np.linalg.norm(position[0] - testing) ** 1
        w2 = 1 / np.linalg.norm(position[1] - testing) ** 1
        solution = (1. * w1 + 2. * w2 * np.ones(10)) / (w1 + w2)
        # testing
        np.testing.assert_array_almost_equal(interp.zn, solution)
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

