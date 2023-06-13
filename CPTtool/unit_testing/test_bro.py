# unit test for the bro
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner
from os.path import join, dirname
from rtree import index
import unittest
import sys
sys.path.append("../")
import bro


class TestGeoMorph(unittest.TestCase):
    """Test reading of the GeoMorph index."""

    def test_index_read(self):
        file_idx = join(join(dirname(__file__), '../bro'), 'geomorph')
        gm_index = index.Index(file_idx)  # created by ../shapefiles/gen_geomorph_idx.py

        geomorphs_nl = list(gm_index.intersection(gm_index.bounds, objects="raw"))

        self.assertTrue(isinstance(geomorphs_nl[0][0], type("")))
        self.assertTrue(isinstance(geomorphs_nl[0][1], type({})))
        self.assertEqual(len(geomorphs_nl), 74121)


class TestBroDb(unittest.TestCase):
    """Test creation of index and parsing of database."""

    def test_database_read_circle(self):
        input = {"BRO_data": "../bro/test_v2_0_1.gpkg", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro_gpkg_version(input)

        self.assertEqual(len(cpts["circle"]["data"]), 24)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 0)

    def test_database_read_polygons(self):
        input = {"BRO_data": "../bro/test_v2_0_1.gpkg", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro_gpkg_version(input)

        key = sorted(cpts["polygons"].keys())[0]

        self.assertEqual(len(cpts["polygons"][key]["data"]), 237)
        self.assertTrue("perc" in cpts["polygons"][key])
        self.assertTrue(isinstance(cpts["polygons"][key]["perc"], float))
        self.assertTrue(100. >= cpts["polygons"][key]["perc"] > 0.)


if __name__ == '__main__':  # pragma: no cover
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
