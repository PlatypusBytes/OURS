from unittest import TestCase
import numpy as np
import sys
import unittest
# add the src folder to the path to search for files
sys.path.append('../')
import netcdf


class TestCDF(TestCase):
    def setUp(self):
        self.pwp = netcdf.NetCDF()
        self.pwp.read_cdffile(r"../bro/peilgebieden_jp_250m.nc")
        return

    def test_coord1(self):
        x = 129463
        y = 454045
        self.pwp.query(x, y)

        self.assertAlmostEqual(-2, self.pwp.NAP_water_level)
        return

    def test_coord2(self):
        x = 196703
        y = 478504
        self.pwp.query(x, y)

        self.assertAlmostEqual(-1.73, self.pwp.NAP_water_level)
        return

    def test_coord3(self):
        x = 85649
        y = 444543
        self.pwp.query(x, y)

        self.assertAlmostEqual(-0.43, self.pwp.NAP_water_level)
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

