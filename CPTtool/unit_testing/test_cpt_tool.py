# unit test for the cpt_module

import sys
import numpy as np
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner
from os.path import join, dirname
from rtree import index
import json

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

    def test_analysis_no_data(self):
        # inputs
        file_properties = 'unit_testing_files\\input_Ground_no_data.json'
        methods_cpt = {'radius': 200}
        output = 'unit_testing_files\\'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)
        properties.close()

        # the function
        cpt_tool.analysis(prop, methods_cpt, output, plots)

        # test if files are filled in correctly
        with open(output + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
            self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                            '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
            self.assertTrue(logfilelines[1], '# Error # : No data in this coordinate point')
            self.assertTrue(logfilelines[2], '# Info # : Analysis finished for coordinate point: ' +
                            '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')' )
        logfile.close()

        # test if the second point log is created correctly
        with open(output + 'log_file_1.txt') as logfile:
            logfilelines = logfile.readlines()
            self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                            '(' + str(prop["Source_x"][1]) + ',' + str(prop["Source_y"][1]) + ')')
            self.assertTrue(logfilelines[1], '# Error # : No data in this coordinate point')
            self.assertTrue(logfilelines[2], '# Info # : Analysis finished for coordinate point: ' +
                            '(' + str(prop["Source_x"][1]) + ',' + str(prop["Source_y"][1]) + ')' )
        logfile.close()

    def test_analysis_cpt_results(self):
        # inputs
        file_properties = 'unit_testing_files\\input_Ground.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 200}
        output = 'unit_testing_files\\'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)
        properties.close()

        # the function
        cpt_tool.analysis(prop, methods_cpt, output, plots)

        # list of cpts
        cptlist = ['CPT000000067109', 'CPT000000065555', 'CPT000000065467', 'CPT000000018872', 'CPT000000067108',
                   'CPT000000065554', 'CPT000000065465', 'CPT000000065466', 'CPT000000065464', 'CPT000000063883',
                   'CPT000000070178', 'CPT000000063847', 'CPT000000063846', 'CPT000000063845', 'CPT000000065410',
                   'CPT000000063882', 'CPT000000062446', 'CPT000000065463', 'CPT000000070146', 'CPT000000065462',
                   'CPT000000065461']

        # read results
        with open(output + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
        logfile.close()
        countercpt = 0
        for counter, line in enumerate(logfilelines):
            if counter == 1:
                self.assertTrue(line, '# Info # : Analysis started for coordinate point: ' +
                                '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
            else:
                if counter == len(logfilelines):
                    self.assertTrue(line, '# Info # : Analysis finished for coordinate point: ' +
                                    '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
                else:
                    if (counter % 2) == 0:
                        self.assertTrue(line, '# Info # : Reading CPT: ' + cptlist[countercpt])
                    else:
                        self.assertTrue(line, '# Info # : Analysis succeeded for: ' + cptlist[countercpt])
                        countercpt = countercpt + 1








    def tearDown(self):
        return


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
