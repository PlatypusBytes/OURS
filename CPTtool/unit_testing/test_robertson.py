from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')
import robertson


class TestRobertson(TestCase):
    def setUp(self):
        return

    def test_soil_types1(self):
        sys.path.append('../../Auxiliary_Code/')
        from shape_file_creator import Create_Shape_File

        actual = Create_Shape_File
        actual.soil_types_robertson(actual)
        test = robertson.Robertson()
        test.soil_types(path_shapefile=r'../shapefiles/')
        np.testing.assert_array_equal(actual.poligon_1 , test.poligon_1)
        np.testing.assert_array_equal(actual.poligon_2,  test.poligon_2)
        np.testing.assert_array_equal(actual.poligon_3 , test.poligon_3)
        np.testing.assert_array_equal(actual.poligon_4,  test.poligon_4)
        np.testing.assert_array_equal(actual.poligon_5 , test.poligon_5)
        np.testing.assert_array_equal(actual.poligon_6,  test.poligon_6)
        np.testing.assert_array_equal(actual.poligon_7 , test.poligon_7)
        np.testing.assert_array_equal(actual.poligon_8,  test.poligon_8)
        np.testing.assert_array_equal(actual.poligon_9,  test.poligon_9)
        return

    def test_lithology(self):

        Rb = robertson.Robertson()
        Rb.soil_types()
        coords_test = []
        Qtn = [2, 2, 10, 7, 20, 100, 900, 700, 700]
        Fr = [0.2, 9, 8, 1, 0.2, 0.5, 0.2, 3, 9]
        lithology_test = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        [coords_test.append([Fr[i], Qtn[i]]) for i in range(len(Fr))]

        litho, coords = robertson.Robertson.lithology(Rb,Qtn, Fr)
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
