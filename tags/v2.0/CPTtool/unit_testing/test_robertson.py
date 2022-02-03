from unittest import TestCase
import numpy as np
import unittest
import sys
sys.path.append("../")
import robertson
sys.path.append("../../Auxiliary_Code")
from shape_file_creator import Create_Shape_File


class TestRobertson(TestCase):
    def setUp(self):
        return

    def test_soil_types1(self):

        actual = Create_Shape_File
        actual.soil_types_robertson(actual)
        actual_list = [actual.poligon_1, actual.poligon_2, actual.poligon_3, actual.poligon_4, actual.poligon_5,
                       actual.poligon_6, actual.poligon_7, actual.poligon_8, actual.poligon_9]
        test = robertson.Robertson()
        test.soil_types()
        counter = 0
        for testpolygon in test.polygons:
            np.testing.assert_array_equal(testpolygon, actual_list[counter])
            counter += + 1
        return

    def test_lithology(self):
        import robertson
        rb = robertson.Robertson()
        rb.soil_types()
        coords_test = []
        Qtn = [2, 2, 10, 7, 20, 100, 900, 700, 700]
        Fr = [0.2, 9, 8, 1, 0.2, 0.5, 0.2, 3, 9]
        lithology_test = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        [coords_test.append([Fr[i], Qtn[i]]) for i in range(len(Fr))]

        litho, coords = robertson.Robertson.lithology(rb, Qtn, Fr)
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
