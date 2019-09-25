from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')
import tools_utils as tu
import os


class TestUtils(TestCase):
    def setUp(self):
        return

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

    def test_add_json(self):

        jsn = {"scenarios": []}
        i = 0

        data = {"IC": [0.1, 0.1, 0.1, 0.1],
                "depth": np.linspace(0, 4, 5),
                "gamma": np.ones(5),
                "G0": np.ones(5),
                "vs": np.ones(5),
                "poisson": np.full(5, 0.3),
                "rho": np.full(5, 3),
                "damping": np.full(5, 3),
                "IC_var": np.ones(4),
                "G0_var": np.zeros(4),
                "poisson_var": np.zeros(4),
                "rho_var": np.zeros(4),
                "damping_var": np.zeros(4),
        }

        depth_json = [0, 1, 2, 3]
        litho_json = ['1.0', '1.0', '1.0', '1.0']
        indx_json = range(5)

        jsn = tu.add_json(jsn, i, depth_json, indx_json, litho_json, data)

        # check if coordinates have been added
        self.assertEqual(jsn['scenarios'][0]['data']['lithology'], litho_json)
        self.assertEqual(jsn['scenarios'][0]['data']['depth'], depth_json)

        # Check if they are equal with the analytical young's modulus
        E = 2 * data["G0"] * (1 + data["poisson"])
        self.assertEqual(jsn['scenarios'][0]['data']['E'], list(map(int, np.round(E.tolist()[:-1]))))
        self.assertEqual(jsn['scenarios'][0]['data']['v'], list(data["poisson"][:-1]))
        self.assertEqual(jsn['scenarios'][0]['data']['rho'], list(map(int, np.round(data["rho"][:-1]))))
        self.assertEqual(jsn['scenarios'][0]['data']['depth'], list(map(int, np.round(data["depth"][:-1]))))
        self.assertEqual(jsn['scenarios'][0]['data']['damping'],  list(data["damping"][:-1]))
        self.assertEqual(jsn['scenarios'][0]['data']['var_E'], list(np.zeros(4)))
        self.assertEqual(jsn['scenarios'][0]['data']['var_v'], list(np.zeros(4)))
        self.assertEqual(jsn['scenarios'][0]['data']['var_rho'], list(np.zeros(4)))
        self.assertEqual(jsn['scenarios'][0]['data']['var_damping'], list(np.zeros(4)))
        self.assertEqual(jsn['scenarios'][0]['data']['var_depth'], list(np.ones(4) * 10.))
        return

    def test_dump_json(self):
        # Set the inputs for the json file
        jsn = {"scenarios": []}
        jsn["scenarios"].append({"coordinates": [1, 2]})

        # Output the json file
        tu.dump_json(jsn, 0, "./")

        # check if file has been created
        self.assertTrue(os.path.isfile("./results_0.json"))

        # Remove file from the directory
        os.remove("./results_0.json")
        return

    def test_interpolation_1(self):
        """
        Single point. interpolation is the same as input. variance nan
        """
        # create cpt object
        cpt = Object()
        cpt.depth = np.linspace(0, 10, 10)
        cpt.NAP = np.linspace(0, -10, 10)
        cpt.tip = np.zeros(10)
        cpt.Qtn = np.ones(10)
        cpt.Fr = np.ones(10) * 2
        cpt.G0 = np.ones(10) * 10
        cpt.poisson = np.ones(10) * 0.12
        cpt.rho = np.ones(10)
        cpt.damping = np.ones(10) * 0.01
        cpt.IC = np.ones(10) * 0.25
        cpt.coord = [10, 20]

        coordinates = ['5', '10']

        cpts_data = {"one": cpt}
        results = tu.interpolation(cpts_data, [coordinates[0], coordinates[1]])

        # test interpolation
        np.testing.assert_array_almost_equal(results["Qtn"], np.ones(10))
        np.testing.assert_array_almost_equal(results["Fr"], np.ones(10) * 2)
        np.testing.assert_array_almost_equal(results["G0"], np.ones(10) * 10)
        np.testing.assert_array_almost_equal(results["poisson"], np.ones(10) * 0.12)
        np.testing.assert_array_almost_equal(results["rho"], np.ones(10))
        np.testing.assert_array_almost_equal(results["damping"], np.ones(10) * 0.01)
        np.testing.assert_array_almost_equal(results["IC"], np.ones(10) * 0.25)
        np.testing.assert_array_almost_equal(results["NAP"], np.linspace(0, -10, 10))
        np.testing.assert_array_almost_equal(results["depth"], np.linspace(0, 10, 10))
        # test variance
        np.testing.assert_array_almost_equal(results["Qtn_var"], np.full(10, np.nan))
        np.testing.assert_array_almost_equal(results["Fr_var"], np.full(10, np.nan))
        np.testing.assert_array_almost_equal(results["G0_var"], np.full(10, np.nan))
        np.testing.assert_array_almost_equal(results["poisson_var"], np.full(10, np.nan))
        np.testing.assert_array_almost_equal(results["rho_var"], np.full(10, np.nan))
        np.testing.assert_array_almost_equal(results["damping_var"], np.full(10, np.nan))
        np.testing.assert_array_almost_equal(results["IC_var"], np.full(10, np.nan))
        return

    def test_interpolation_2(self):
        """
        Two points. interpolation is the middle point.
        """
        # create cpt object
        cpt1 = Object()
        cpt1.depth = np.linspace(0, 10, 10)
        cpt1.NAP = np.linspace(0, -10, 10)
        cpt1.tip = np.zeros(10)
        cpt1.Qtn = np.ones(10)
        cpt1.Fr = np.ones(10) * 2
        cpt1.G0 = np.ones(10) * 10
        cpt1.poisson = np.ones(10) * 0.12
        cpt1.rho = np.ones(10)
        cpt1.damping = np.ones(10) * 0.01
        cpt1.IC = np.ones(10) * 0.25
        cpt1.coord = [10, 20]

        cpt2 = Object()
        cpt2.depth = np.linspace(1, 11, 10)
        cpt2.NAP = np.linspace(-1, -11, 10)
        cpt2.tip = np.zeros(10) * 2
        cpt2.Qtn = np.ones(10) * 2
        cpt2.Fr = np.ones(10) * 2 * 2
        cpt2.G0 = np.ones(10) * 10 * 2
        cpt2.poisson = np.ones(10) * 0.12 * 2
        cpt2.rho = np.ones(10) * 2
        cpt2.damping = np.ones(10) * 0.01 * 2
        cpt2.IC = np.ones(10) * 0.25 * 2
        cpt2.coord = [20, 40]

        coordinates = ['15', '30']

        cpts_data = {"one": cpt1, "two": cpt2}
        results = tu.interpolation(cpts_data, [coordinates[0], coordinates[1]])

        # test interpolation
        np.testing.assert_array_almost_equal(results["Qtn"], np.ones(10) * 1.5)
        np.testing.assert_array_almost_equal(results["Fr"], np.ones(10) * 2 * 1.5)
        np.testing.assert_array_almost_equal(results["G0"], np.ones(10) * 10 * 1.5)
        np.testing.assert_array_almost_equal(results["poisson"], np.ones(10) * 0.12 * 1.5)
        np.testing.assert_array_almost_equal(results["rho"], np.ones(10) * 1.5)
        np.testing.assert_array_almost_equal(results["damping"], np.ones(10) * 0.01 * 1.5)
        np.testing.assert_array_almost_equal(results["IC"], np.ones(10) * 0.25 * 1.5)
        # np.testing.assert_array_almost_equal(results["NAP"], np.linspace(-0.5, -10.5, 10))
        np.testing.assert_array_almost_equal(results["depth"], np.linspace(0.5, 10.5, 10) - 0.5)
        # test variance
        np.testing.assert_array_almost_equal(results["Qtn_var"], np.var([cpt1.Qtn, cpt2.Qtn]))
        np.testing.assert_array_almost_equal(results["Fr_var"], np.var([cpt1.Fr, cpt2.Fr]))
        np.testing.assert_array_almost_equal(results["G0_var"], np.var([cpt1.G0, cpt2.G0]))
        np.testing.assert_array_almost_equal(results["poisson_var"], np.var([cpt1.poisson, cpt2.poisson]))
        np.testing.assert_array_almost_equal(results["rho_var"], np.var([cpt1.rho, cpt2.rho]))
        np.testing.assert_array_almost_equal(results["damping_var"], np.var([cpt1.damping, cpt2.damping]))
        np.testing.assert_array_almost_equal(results["IC_var"], np.var([cpt1.IC, cpt2.IC]))
        return

    def test_interpolation_3(self):
        """
        Two points. interpolation is the first point.
        """
        # create cpt object
        cpt1 = Object()
        cpt1.depth = np.linspace(0, 10, 10)
        cpt1.NAP = np.linspace(0, -10, 10)
        cpt1.tip = np.zeros(10)
        cpt1.Qtn = np.ones(10)
        cpt1.Fr = np.ones(10) * 2
        cpt1.G0 = np.ones(10) * 10
        cpt1.poisson = np.ones(10) * 0.12
        cpt1.rho = np.ones(10)
        cpt1.damping = np.ones(10) * 0.01
        cpt1.IC = np.ones(10) * 0.25
        cpt1.coord = [10, 20]

        cpt2 = Object()
        cpt2.depth = np.linspace(1, 11, 10)
        cpt2.NAP = np.linspace(-1, -11, 10)
        cpt2.tip = np.zeros(10) * 2
        cpt2.Qtn = np.ones(10) * 2
        cpt2.Fr = np.ones(10) * 2 * 2
        cpt2.G0 = np.ones(10) * 10 * 2
        cpt2.poisson = np.ones(10) * 0.12 * 2
        cpt2.rho = np.ones(10) * 2
        cpt2.damping = np.ones(10) * 0.01 * 2
        cpt2.IC = np.ones(10) * 0.25 * 2
        cpt2.coord = [20, 40]

        coordinates = ['11', '21']

        cpts_data = {"one": cpt1, "two": cpt2}
        results = tu.interpolation(cpts_data, [coordinates[0], coordinates[1]])

        # test interpolation
        d1 = np.sqrt((11 - 10) ** 2 + (21 - 20) ** 2)
        d2 = np.sqrt((11 - 20) ** 2 + (21 - 40) ** 2)

        w1 = (1 / d1) / (1 / d1 + 1/ d2)
        w2 = (1 / d2) / (1 / d1 + 1/ d2)

        np.testing.assert_array_almost_equal(results["Qtn"], cpt1.Qtn * w1 + cpt2.Qtn * w2)
        np.testing.assert_array_almost_equal(results["Fr"], cpt1.Fr * w1 + cpt2.Fr * w2)
        np.testing.assert_array_almost_equal(results["G0"], cpt1.G0 * w1 + cpt2.G0 * w2)
        np.testing.assert_array_almost_equal(results["poisson"], cpt1.poisson * w1 + cpt2.poisson * w2)
        np.testing.assert_array_almost_equal(results["rho"], cpt1.rho * w1 + cpt2.rho * w2)
        np.testing.assert_array_almost_equal(results["damping"], cpt1.damping * w1 + cpt2.damping * w2)
        np.testing.assert_array_almost_equal(results["IC"], cpt1.IC * w1 + cpt2.IC * w2)
        np.testing.assert_array_almost_equal(np.round(results["NAP"]), np.round(cpt1.NAP * w1 + cpt2.NAP * w2))
        np.testing.assert_array_almost_equal(results["depth"], (cpt1.depth * w1 + cpt2.depth * w2) - (cpt1.depth * w1 + cpt2.depth * w2)[0])
        # test variance
        mean = np.mean(cpt1.Qtn * w1 + cpt2.Qtn * w2)
        var = (cpt1.Qtn - mean) ** 2 * w1 + (cpt2.Qtn - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["Qtn_var"], var)
        mean = np.mean(cpt1.Fr * w1 + cpt2.Fr * w2)
        var = (cpt1.Fr - mean) ** 2 * w1 + (cpt2.Fr - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["Fr_var"], var)
        mean = np.mean(cpt1.G0 * w1 + cpt2.G0 * w2)
        var = (cpt1.G0 - mean) ** 2 * w1 + (cpt2.G0 - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["G0_var"], var)
        mean = np.mean(cpt1.poisson * w1 + cpt2.poisson * w2)
        var = (cpt1.poisson - mean) ** 2 * w1 + (cpt2.poisson - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["poisson_var"], var)
        mean = np.mean(cpt1.rho * w1 + cpt2.rho * w2)
        var = (cpt1.rho - mean) ** 2 * w1 + (cpt2.rho - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["rho_var"], var)
        mean = np.mean(cpt1.damping * w1 + cpt2.damping * w2)
        var = (cpt1.damping - mean) ** 2 * w1 + (cpt2.damping - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["damping_var"], var)
        mean = np.mean(cpt1.IC * w1 + cpt2.IC * w2)
        var = (cpt1.IC - mean) ** 2 * w1 + (cpt2.IC - mean) ** 2 * w2
        np.testing.assert_array_almost_equal(results["IC_var"], var)
        return

    def tearDown(self):
        return


class Object(object):
    pass


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity
    from teamcity.unittestpy import TeamcityTestRunner
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)

