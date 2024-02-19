# unit test for the bro
from os.path import join, dirname
from rtree import index
from shapely.geometry import Point
import unittest
import shutil
import sqlite3
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



class TestBufferedRail(unittest.TestCase):
    """Test reading of the GeoMorph index."""

    def test_index_read(self):
        file_idx = join('bro', 'buff_track')
        file_index = index.Index(file_idx)  # created by ../shapefiles/gen_buffer_track_idx.py

        list_points = [[129125.375, 471341],
                    [128894, 470670],
                    [127877, 464928],
                    [131565.523, 471481.448], # False
                    [88712.725, 451981.087],
                    [88548, 452886], # False
                    [93744.64, 460834.3], # False
                    [82825.863, 455054.337]
                    ]

        result = []
        for point in list_points:
            point = Point(point[0], point[1])  # Create a shapely Point
            possible_matches = list(file_index.intersection(point.bounds))
            if len(possible_matches) == 0:
                result.append(False)
            else:
                result.append(True)

        self.assertTrue(result, [True, True, True, False, True, False, False, True])

    def test_function(self):
        file_idx = join('bro', 'buff_track')

        list_points = [[129125.375, 471341],
                    [128894, 470670],
                    [127877, 464928],
                    [131565.523, 471481.448], # False
                    [88712.725, 451981.087],
                    [88548, 452886], # False
                    [93744.64, 460834.3], # False
                    [82825.863, 455054.337]
                    ]

        result = []
        for point in list_points:
            point = [point[0], point[1]]  # Create a shapely Point
            result.append(bro.is_cpt_inside_buffered_track(file_idx, point))

        self.assertTrue(result, [True, True, True, False, True, False, False, True])


class TestBroDb(unittest.TestCase):
    """Test creation of index and parsing of database."""

    def test_database_read_circle(self):
        # get absolute path to the test database
        test_db = join(dirname(__file__), '../bro/test_v2_0_1.gpkg')
        input = {"BRO_data": test_db, "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro_gpkg_version(input)

        self.assertEqual(len(cpts["circle"]["data"]), 24)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 0)

    def test_database_read_polygons(self):
        # get absolute path to the test database
        test_db = join(dirname(__file__), '../bro/test_v2_0_1.gpkg')
        input = {"BRO_data": test_db, "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro_gpkg_version(input)

        key = sorted(cpts["polygons"].keys())[0]

        self.assertEqual(len(cpts["polygons"][key]["data"]), 237)
        self.assertTrue("perc" in cpts["polygons"][key])
        self.assertTrue(isinstance(cpts["polygons"][key]["perc"], float))
        self.assertTrue(100. >= cpts["polygons"][key]["perc"] > 0.)

    def test_database_indexes_created(self):
        # get absolute path to the test database
        test_db = join(dirname(__file__), '../bro/test_v2_0_2_idx.gpkg')
        test_db_new = join(dirname(__file__), '../bro/test_v2_0_2_idx_test.gpkg')
        shutil.copyfile(test_db, test_db_new)


        # check that indexes does not exist
        check_query = "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_test1'"
        conn = sqlite3.connect(test_db_new, uri=True)
        cursor = conn.cursor()
        cursor.execute(check_query)
        index_exists = cursor.fetchone() is not None
        conn.close()

        # check that indexes are created
        self.assertFalse(index_exists)

        # create indexes
        bro.create_index_gpkg(test_db_new)
        check_query = "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_test1'"
        conn = sqlite3.connect(test_db_new, uri=True)
        cursor = conn.cursor()
        cursor.execute(check_query)
        index_exists = cursor.fetchone() is not None
        conn.close()

        self.assertTrue(index_exists)

        import os
        # remove the test database
        os.remove(test_db_new)
