# unit test for the cpt_module

import sys
import numpy as np
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner
from os.path import join, dirname
from rtree import index

# add the src folder to the path to search for files
sys.path.append('../')
import unittest
import cpt_tool
import bro

class TestCptTool(unittest.TestCase):
    def setUp(self):
        return

    # def test_set_key(self):
    #     local_labels = ['depth', 'tip', 'friction', 'friction_nb', 'water']
    #     local_dat = [1, 2, 3, 4, 6]
    #     key = cpt_tool.set_key()
    #     for i in range(len(local_dat)):
    #         np.testing.assert_equal(key[local_labels[i]],local_dat[i])
    #     return

    def tearDown(self):
        return


class TestGeoMorph(unittest.TestCase):
    """Test reading of the GeoMorph index."""

    def test_index_read(self):
        file_idx = join(join(dirname(__file__), '../shapefiles'), 'geomorph')
        gm_index = index.Index(file_idx)  # created by ../shapefiles/gen_geomorph_idx.py

        geomorphs_nl = list(gm_index.intersection(gm_index.bounds, objects="raw"))

        self.assertTrue(isinstance(geomorphs_nl[0][0], type("")))
        self.assertTrue(isinstance(geomorphs_nl[0][1], type({})))
        self.assertEqual(len(geomorphs_nl), 74121)


class TestBroDb(unittest.TestCase):
    """Test creation of index and parsing of database."""

    def test_database_read_circle(self):
        input = {"BRO_data": "../bro/brocpt.xml", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro(input)

        self.assertEqual(len(cpts["circle"]["data"]), 23)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 2)

    def test_database_read_polygons(self):
        input = {"BRO_data": "../bro/brocpt.xml", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro(input)

        key = sorted(cpts["polygons"].keys())[0]

        self.assertEqual(len(cpts["polygons"][key]["data"]), 23)
        self.assertTrue("perc" in cpts["polygons"][key])
        self.assertTrue(isinstance(cpts["polygons"][key]["perc"], float))
        self.assertTrue(100. >= cpts["polygons"][key]["perc"] > 0.)

        self.assertEqual(len(cpts["polygons_nl"][key]["data"]), 23)
        self.assertTrue("count" in cpts["polygons_nl"][key])
        self.assertTrue(isinstance(cpts["polygons_nl"][key]["count"], int))
        self.assertTrue(cpts["polygons_nl"][key]["count"] >= 1)

    def test_zipdatabase_read(self):
        input = {"BRO_data": "../bro/brocpt.zip", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro(input)

        self.assertEqual(len(cpts["circle"]["data"]), 23)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 2)


if __name__ == '__main__':  # pragma: no cover
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
