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
        self.cpt = cpt_module.CPT()
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
    def tearDown(self):
        return


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity
    if is_running_under_teamcity():
        from teamcity.unittestpy import TeamcityTestRunner
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)