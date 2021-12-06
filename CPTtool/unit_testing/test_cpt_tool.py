# unit test for the cpt_module
import numpy as np
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner
from os.path import join, dirname
from rtree import index
import json
import unittest
from CPTtool import cpt_tool
from CPTtool import bro
from CPTtool import log_handler


class TestCptTool(unittest.TestCase):
    def setUp(self):
        # settings
        self.settings = {"minimum_length": 5,  # minimum length of CPT
                         "minimum_samples": 50,  # minimum number of samples of CPT
                         "minimum_ratio": 0.1,  # mimimum ratio of correct values in a CPT
                         "convert_to_kPa": True,  # convert CPT to kPa
                         "nb_points": 5,  # number of points for smoothing
                         "limit": 0,  # lower bound of the smooth function
                         "gamma_min": 10.5,  # minimum unit weight
                         "gamma_max": 22,  # maximum unit weight
                         "d_min": 2.,  # parameter for damping (minimum damping)
                         "Cu": 2.,  # parameter for damping (coefficient of uniformity)
                         "D50": 0.2,  # parameter for damping (median grain size)
                         "Ip": 40.,  # parameter for damping (plastic index)
                         "freq": 1.,  # parameter for damping (frequency)
                         "lithologies": ["1", "2"],  # lithologies to filter
                         "key": "G0",  # attribute to filder
                         "value": 1e6,  # lower value to filter
                         "power": 1,  # power for IDW interpolation
                         }
        return

    def test_define_settings_error(self):
        with self.assertRaises(SystemExit):
            cpt_tool.define_settings('fake_file.json')
        return

    def test_define_settings_no_file(self):
        sett = cpt_tool.define_settings(False)
        for s in sett:
            self.assertEqual(sett[s], self.settings[s])
        return

    def test_define_settings(self):
        sett = cpt_tool.define_settings(r'./unit_testing_files/settings.json')

        settings = {"minimum_length": 5,  # minimum length of CPT
                    "minimum_samples": 50,  # minimum number of samples of CPT
                    "minimum_ratio": 0.1,  # mimimum ratio of correct values in a CPT
                    "convert_to_kPa": False,  # convert CPT to kPa
                    "nb_points": 5,  # number of points for smoothing
                    "limit": 0,  # lower bound of the smooth function
                    "gamma_min": 10.5,  # minimum unit weight
                    "gamma_max": 220,  # maximum unit weight
                    "d_min": 2.,  # parameter for damping (minimum damping)
                    "Cu": 2.,  # parameter for damping (coefficient of uniformity)
                    "D50": 0.2,  # parameter for damping (median grain size)
                    "Ip": 40.,  # parameter for damping (plastic index)
                    "freq": 1.,  # parameter for damping (frequency)
                    "lithologies": ["1", "2", "5"],  # lithologies to filter
                    "key": "poisson",  # attribute to filder
                    "value": 1e6,  # lower value to filter
                    "power": 10,  # power for IDW interpolation
                    }
        for s in sett:
            self.assertEqual(sett[s], settings[s])
        return

    def test_define_methods_no_keys_in_file(self):
        with self.assertRaises(SystemExit):
            cpt_tool.define_methods(r'unit_testing_files\\settings_no_key.json')
        return

    def test_define_methods_error(self):
        with self.assertRaises(SystemExit):
            cpt_tool.define_methods('fake_file.json')
        return

    def test_define_methods_no_file(self):
        methods = cpt_tool.define_methods(None)
        self.assertEqual(methods['gamma'], "Lengkeek")
        self.assertEqual(methods['vs'], "Mayne")
        self.assertEqual(methods['OCR'], "Mayne")
        self.assertEqual(methods['radius'], 600)
        return

    def test_define_methods(self):
        methods = cpt_tool.define_methods('unit_testing_files\\methods.json')
        self.assertEqual(methods['gamma'], "Lengkeek")
        self.assertEqual(methods['vs'], "Andrus")
        self.assertEqual(methods['OCR'], "Mayne")
        self.assertEqual(methods['radius'], 1e10)
        return

    def test_define_methods_no_keys_in_file(self):
        with self.assertRaises(SystemExit):
            cpt_tool.define_methods('unit_testing_files\\methods_no_keys.json')

        with self.assertRaises(SystemExit):
            cpt_tool.define_methods('unit_testing_files\\methods_gamma_missing.json')

        with self.assertRaises(SystemExit):
            cpt_tool.define_methods('unit_testing_files\\methods_vs_missing.json')

        with self.assertRaises(SystemExit):
            cpt_tool.define_methods('unit_testing_files\\methods_OCR_missing.json')

        with self.assertRaises(SystemExit):
            cpt_tool.define_methods('unit_testing_files\\methods_error_radius.json')
        return

    def test_read_json(self):
        with self.assertRaises(SystemExit):
            cpt_tool.read_json('fake_file.json')

        data = cpt_tool.read_json('unit_testing_files\\input_Ground.json')
        self.assertTrue(data != {})
        return

    def test_parse_cpt(self):
        # inputs
        file_properties = 'unit_testing_files\\input_Ground.json'
        methods_cpt = {'radius': 100}
        with open(file_properties) as properties:
            prop = json.load(properties)

        # read BRO data base
        inpt = {"BRO_data": prop["BRO_data"],
                "Source_x": float(prop["Source_x"][0]), "Source_y": float(prop["Source_y"][0]),
                "Radius": float(methods_cpt["radius"])}

        cpt_BRO = bro.read_bro(inpt)
        dataframe = cpt_BRO['polygons']['2M81ykd']['data'][0]['dataframe']

        # define target columns and values
        target_columns = ['penetrationLength', 'depth', 'elapsedTime', 'coneResistance', 'inclinationEW',
                          'inclinationNS', 'inclinationResultant', 'localFriction', 'porePressureU2', 'frictionRatio']

        target_values = [1.105, 1.105, 242.800, 0.160, -1.000,
                         0.000, 1.000, 0.001, 0.001, np.nan]

        # assert if only the columns are read which should be read,
        # assert if the values in the first row of the dataframe are correct
        for idx, column in enumerate(dataframe.columns.values):
            self.assertEqual(column, target_columns[idx])
            if np.isnan(target_values[idx]):
                self.assertTrue(np.isnan(dataframe.values[0][idx]))
            else:
                self.assertAlmostEqual(dataframe.values[0][idx], target_values[idx], places=3)

    def test_read_cpt(self):
        # inputs
        file_properties = 'unit_testing_files\\input_Ground.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 100}
        output = 'unit_testing_files\\results\\'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)

        # read BRO data base
        inpt = {"BRO_data": prop["BRO_data"],
                "Source_x": float(prop["Source_x"][0]), "Source_y": float(prop["Source_y"][0]),
                "Radius": float(methods_cpt["radius"])}
        cpt_BRO = bro.read_bro(inpt)

        log_file = log_handler.LogFile(output, 0)
        cpt_BRO['polygons']['2M81ykd']['data'][0]['dataframe'].depth= \
            cpt_BRO['polygons']['2M81ykd']['data'][0]['dataframe'].depth.dropna().empty
        cpt_BRO['polygons']['2M81ykd']['data'][1]['dataframe'].depth = \
            cpt_BRO['polygons']['2M81ykd']['data'][1]['dataframe'].depth.dropna().empty
        data = list(filter(None, cpt_BRO['polygons']['2M81ykd']['data']))

        jsn, is_jsn_modified = cpt_tool.read_cpt(data, methods_cpt, self.settings, output,
                                                 {"Receiver_x": prop["Source_x"][0], "Receiver_y": prop["Source_y"][0],
                                                  "MinLayerThickness": '0.5', "BRO_data": prop["BRO_data"]},
                                                 plots, 0, log_file, {"scenarios": []}, 0)

        # check json file
        self.assertTrue(is_jsn_modified)
        self.assertTrue(bool(jsn))
        return

    def test_read_cpt_empty(self):
        # In this test all the cpts do not have good quality data that means that no results will be returned.
        # So an empty json file with a False is_jsn_modified statement
        # inputs
        import pandas as pd
        file_properties = 'unit_testing_files\\input_Ground.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 100}
        output = 'unit_testing_files\\results\\'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)

        inpt = {"BRO_data": prop["BRO_data"],
                "Source_x": float(prop["Source_x"][0]), "Source_y": float(prop["Source_y"][0]),
                "Radius": float(methods_cpt["radius"])}

        log_file = log_handler.LogFile(output, 0)

        d = {'penetrationLength': [0.1, 0.2],
             'coneResistance': [1, 2],
             'localFriction': [3, 4],
             'porePressureU2': [0.5, 1],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    "a": 0.85,
                    "dataframe": df,
                    'predrilled_z': 0.}
        data = [cpt_data]

        jsn, is_jsn_modified = cpt_tool.read_cpt(data, methods_cpt, self.settings, output,
                                                 {"Receiver_x": prop["Source_x"][0], "Receiver_y": prop["Source_y"][0],
                                                  "MinLayerThickness": '0.5', "BRO_data": prop["BRO_data"]},
                                                 plots, 0, log_file, {"scenarios": []}, 0)

        # check json file
        self.assertFalse(is_jsn_modified)
        self.assertTrue(jsn == {'scenarios': []})
        return

    def test_analysis_no_data(self):
        # inputs
        file_properties = 'unit_testing_files\\input_Ground_no_data.json'
        methods_cpt = {'radius': 200}
        output = 'unit_testing_files\\results'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)

        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # test if files are filled in correctly
        with open(output + '\\' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
            self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                            '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
            self.assertTrue(logfilelines[1], '# Error # : No data in this coordinate point')
            self.assertTrue(logfilelines[2], '# Info # : Analysis finished for coordinate point: ' +
                            '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')' )

        # test if the second point log is created correctly
        with open(output + '\\' + 'log_file_1.txt') as logfile:
            logfilelines = logfile.readlines()
            self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                            '(' + str(prop["Source_x"][1]) + ',' + str(prop["Source_y"][1]) + ')')
            self.assertTrue(logfilelines[1], '# Error # : No data in this coordinate point')
            self.assertTrue(logfilelines[2], '# Info # : Analysis finished for coordinate point: ' +
                            '(' + str(prop["Source_x"][1]) + ',' + str(prop["Source_y"][1]) + ')' )

    def test_analysis_only_circles(self):
        import os
        # inputs
        file_properties = 'unit_testing_files\\input_Ground_only_circle.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 100}
        output = 'unit_testing_files\\results'
        plots = True
        with open(file_properties) as properties:
            prop = json.load(properties)

        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # the results cpts
        cpt_results = ['CPT000000000448', 'CPT000000000449']

        # read results
        with open(output + '\\' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
        countercpt = 0

        # check log file
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
                        self.assertTrue(line, '# Info # : Reading CPT: ' + cpt_results[countercpt])
                    else:
                        self.assertTrue(line, '# Info # : Analysis succeeded for: ' + cpt_results[countercpt])
                        countercpt = countercpt + 1

        # check json file
        with open(output + '\\' + 'results_0.json') as jsonfile:
            jsonresults = json.load(jsonfile)

        self.assertEqual(jsonresults['scenarios'][0]['probability'], 1.)

        # Check plots
        for i in cpt_results:
            self.assertTrue(os.path.exists(output + '\\' + i + '.csv'))
            self.assertTrue(os.path.exists(output + '\\' + i + '_cpt.png'))
            self.assertTrue(os.path.exists(output + '\\' + i + '_lithology.png'))
        return

    def test_analysis_polygon_and_circle(self):
        import os
        # inputs
        file_properties = 'unit_testing_files\\input_Ground.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 100}
        output = 'unit_testing_files\\results'
        plots = True
        with open(file_properties) as properties:
            prop = json.load(properties)

        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # the results cpts
        cpt_results = ['CPT000000000207', 'CPT000000000197', 'CPT000000000191', 'CPT000000000193', 'CPT000000000200',
                       'CPT000000000388', 'CPT000000000201']

        # read results
        with open(output + '\\' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
        countercpt = 0

        # check log file
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
                        self.assertTrue(line, '# Info # : Reading CPT: ' + cpt_results[countercpt])
                    else:
                        self.assertTrue(line, '# Info # : Analysis succeeded for: ' + cpt_results[countercpt])
                        countercpt = countercpt + 1

        # check json file
        with open(output + '\\' + 'results_0.json') as jsonfile:
            jsonresults = json.load(jsonfile)

        print(jsonresults['scenarios'][0]['probability'])
        self.assertEqual(jsonresults['scenarios'][0]['probability'], 0.231)
        self.assertEqual(jsonresults['scenarios'][1]['probability'], 0.769)

        # Check plots
        for i in cpt_results:
            self.assertTrue(os.path.exists(output + '\\' + i + '.csv'))
            self.assertTrue(os.path.exists(output + '\\' + i + '_cpt.png'))
            self.assertTrue(os.path.exists(output + '\\' + i + '_lithology.png'))
        return

    def test_analysis_only_polygon(self):
        import os
        # inputs
        file_properties = 'unit_testing_files\\input_Ground_only_polygon.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 150}
        output = 'unit_testing_files\\results'
        plots = True
        with open(file_properties) as properties:
            prop = json.load(properties)

        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # the results cpts
        cpt_results = ['CPT000000000022', 'CPT000000000023', 'CPT000000000024', 'CPT000000000025', 'CPT000000000029',
                       'CPT000000000030', 'CPT000000000026']

        # read results
        with open(output + '\\' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
        countercpt = 0

        # check log file
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
                        self.assertTrue(line, '# Info # : Reading CPT: ' + cpt_results[countercpt])
                    else:
                        self.assertTrue(line, '# Info # : Analysis succeeded for: ' + cpt_results[countercpt])
                        countercpt = countercpt + 1

        # check json file
        with open(output + '\\' + 'results_0.json') as jsonfile:
            jsonresults = json.load(jsonfile)

        self.assertEqual(jsonresults['scenarios'][0]['probability'], 1.)

        # Check plots
        for i in cpt_results:
            self.assertTrue(os.path.exists(output + '\\' + i + '.csv'))
            self.assertTrue(os.path.exists(output + '\\' + i + '_cpt.png'))
            self.assertTrue(os.path.exists(output + '\\' + i + '_lithology.png'))
        return

    def tearDown(self):
        import shutil
        import os
        if os.path.exists('unit_testing_files\\results'):
            shutil.rmtree('unit_testing_files\\results')
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
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 0)

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
        self.assertEqual(len(list(filter(lambda x: x is None, cpts["circle"]["data"]))), 0)


if __name__ == '__main__':  # pragma: no cover
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
