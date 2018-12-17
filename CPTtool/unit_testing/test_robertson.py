from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')
from CPTtool.robertson import Robertson

class TestRobertson(TestCase):
    def setUp(self):
        from CPTtool.robertson import Robertson
        pass

#    def test_soil_types(self):
#        self.fail()

    def test_lithology(self):
        Rb = Robertson()
        Rb.soil_types()
        coords_test = []
        Qtn = [2, 2, 10, 7, 20, 100, 900, 700, 700]
        Fr = [0.2, 9, 8, 1, 0.2, 0.5, 0.2, 3, 9]
        lithology_test = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        [coords_test.append([Fr[i], Qtn[i]]) for i in range(len(Fr))]

        litho, coords = Robertson.lithology(Rb,Qtn, Fr)
        np.testing.assert_array_equal(coords_test, coords)
        np.testing.assert_array_equal(lithology_test, litho)
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