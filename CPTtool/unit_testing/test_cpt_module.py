# unit test for the cpt_module

import sys
# add the src folder to the path to search for files
sys.path.append('../')
import unittest
import cpt_module
import numpy as np


class TestCptModule(unittest.TestCase):
    def setUp(self):
        import cpt_module
        self.cpt = cpt_module.CPT("./")
        pass

    def test_rho_calculation(self):
        self.cpt.gamma = np.ones(10)
        self.cpt.g = 10.
        self.cpt.rho_calc()

        # exact solution = gamma / g
        exact_rho = np.ones(10) * 1000 / 10

        # self.assertEqual(exact_rho, self.cpt.rho)
        np.testing.assert_array_equal(exact_rho, self.cpt.rho)
        return

    def test_gamma_calc(self):
        gamma_limit = 22
        self.cpt.friction_nbr = np.ones(10)
        self.cpt.qt = np.ones(10)
        self.cpt.Pa = 100
        # Exact solution
        np.seterr(divide="ignore")
        aux = 0.27*np.log10(np.ones(10))+0.36*(np.log10(np.ones(10)/ 100))+1.236
        aux[np.abs(aux) == np.inf] = gamma_limit / 9.81
        local_gamma = aux * 9.81

        self.cpt.gamma_calc(gamma_limit)

        np.testing.assert_array_equal(local_gamma, self.cpt.gamma)
        return

    def test_merge_thickness_1(self):
        min_layer_thick = 0.5
        self.cpt.depth = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        self.cpt.lithology = ['0','0','0','0','0','1','2','2','2','2','2','2']
        self.cpt.IC = [0.5,0.5,0.5,0.5,0.5,1,3,3,3,3,3,3]
        merged = self.cpt.merge_thickness(min_layer_thick)
        depth_test = [0.0,0.6]
        test_lithology = ['/0/1','/2']
        test_index = [0,6]
        np.testing.assert_array_equal(depth_test, self.cpt.depth_json)
        np.testing.assert_array_equal(test_lithology, self.cpt.lithology_json)
        np.testing.assert_array_equal(test_index, self.cpt.indx_json)

        self.cpt.IC = [3,3,3,3,3,1,0.5,0.5,0.5,0.5,0.5,0.5]
        merged = self.cpt.merge_thickness(min_layer_thick)
        depth_test = [0.0,0.5]
        test_lithology = ['/0','/1/2']
        test_index = [0,5]
        np.testing.assert_array_equal(depth_test, self.cpt.depth_json)
        np.testing.assert_array_equal(test_lithology, self.cpt.lithology_json)
        np.testing.assert_array_equal(test_index, self.cpt.indx_json)

        return

    def tearDown(self):
        return


if __name__ == '__main__':  # pragma: no cover
#    from teamcity import is_running_under_teamcity
#    if is_running_under_teamcity():
#        from teamcity.unittestpy import TeamcityTestRunner
#        runner = TeamcityTestRunner()
#   else:
    runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)