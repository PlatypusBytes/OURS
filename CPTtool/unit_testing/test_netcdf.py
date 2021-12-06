from unittest import TestCase
import unittest
from CPTtool import netcdf


class TestCDF(TestCase):
    def setUp(self):
        self.pwp = netcdf.NetCDF()
        self.pwp.read_cdffile(r"../bro/peilgebieden_jp_250m.nc")
        return

    def test_coord1(self):
        x = 129463
        y = 454045
        self.pwp.query(x, y)

        self.assertAlmostEqual(-1.649999976158142, float(self.pwp.NAP_water_level))
        return

    def test_coord2(self):
        x = 196704
        y = 478504
        self.pwp.query(x, y)

        self.assertAlmostEqual( 2.6500000953674316, float(self.pwp.NAP_water_level))
        return

    def test_coord3(self):
        x = 85649
        y = 444543
        self.pwp.query(x, y)

        self.assertAlmostEqual(-0.4300000071525574, float(self.pwp.NAP_water_level))
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

