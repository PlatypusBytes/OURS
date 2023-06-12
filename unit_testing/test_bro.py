# unit test for the bro
from os.path import join, dirname
from rtree import index
import unittest
from CPTtool import bro


class TestGeoMorph(unittest.TestCase):
    """Test reading of the GeoMorph index."""

    def test_index_read(self):
        file_idx = join('bro', 'geomorph')
        gm_index = index.Index(file_idx)  # created by ../shapefiles/gen_geomorph_idx.py

        geomorphs_nl = list(gm_index.intersection(gm_index.bounds, objects="raw"))

        self.assertTrue(isinstance(geomorphs_nl[0][0], type("")))
        self.assertTrue(isinstance(geomorphs_nl[0][1], type({})))
        self.assertEqual(len(geomorphs_nl), 74121)


class TestBroDb(unittest.TestCase):
    """Test creation of index and parsing of database."""

    def test_database_read_circle(self):
        input = {"BRO_data": "./bro/brocptvolledigeset.gpkg", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro_gpkg_version(input)

        self.assertEqual(len(cpts["circle"]["data"]), 23)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 0)

    def test_database_read_polygons(self):
        input = {"BRO_data": "./bro/brocptvolledigeset.gpkg", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro_gpkg_version(input)

        key = sorted(cpts["polygons"].keys())[0]

        self.assertEqual(len(cpts["polygons"][key]["data"]), 23)
        self.assertTrue("perc" in cpts["polygons"][key])
        self.assertTrue(isinstance(cpts["polygons"][key]["perc"], float))
        self.assertTrue(100. >= cpts["polygons"][key]["perc"] > 0.)


if __name__ == '__main__':
    unittest.main()
