# unit test for the cpt_tool
import numpy as np
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
        # create absolute path to the test settings file
        test_settings = join(dirname(__file__), 'unit_testing_files/settings.json')
        sett = cpt_tool.define_settings(test_settings)

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
        # create absolute path to the test settings file
        test_settings = join(dirname(__file__), 'unit_testing_files/settings_no_key.json')
        with self.assertRaises(SystemExit):
            cpt_tool.define_methods(test_settings)
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
        test_settings = join(dirname(__file__), 'unit_testing_files/methods.json')
        methods = cpt_tool.define_methods(test_settings)
        self.assertEqual(methods['gamma'], "Lengkeek")
        self.assertEqual(methods['vs'], "Andrus")
        self.assertEqual(methods['OCR'], "Mayne")
        self.assertEqual(methods['radius'], 1e10)
        return

    def test_define_methods_no_keys_in_file(self):
        with self.assertRaises(SystemExit):
            test_settings = join(dirname(__file__), 'unit_testing_files/methods_no_keys.json')
            cpt_tool.define_methods(test_settings)

        with self.assertRaises(SystemExit):
            test_settings = join(dirname(__file__), 'unit_testing_files/methods_gamma_missing.json')
            cpt_tool.define_methods(test_settings)

        with self.assertRaises(SystemExit):
            test_settings = join(dirname(__file__), 'unit_testing_files/methods_vs_missing.json')
            cpt_tool.define_methods(test_settings)

        with self.assertRaises(SystemExit):
            test_settings = join(dirname(__file__), 'unit_testing_files/methods_OCR_missing.json')
            cpt_tool.define_methods(test_settings)

        with self.assertRaises(SystemExit):
            test_settings = join(dirname(__file__), 'unit_testing_files/methods_radius_missing.json')
            cpt_tool.define_methods(test_settings)
        return

    def test_read_json(self):
        with self.assertRaises(SystemExit):
            cpt_tool.read_json('fake_file.json')
        test_settings = join(dirname(__file__), 'unit_testing_files/input_Ground.json')
        data = cpt_tool.read_json(test_settings)
        self.assertTrue(data != {})
        return

    def test_parse_cpt(self):
        # inputs
        test_settings = join(dirname(__file__), 'unit_testing_files/input_Ground.json')
        file_properties = test_settings
        methods_cpt = {'radius': 100}
        with open(file_properties) as properties:
            prop = json.load(properties)

        # read BRO data base
        # bro location to absolute path
        bro_location = join(dirname(__file__), "..", prop["BRO_data"])
        inpt = {"BRO_data": bro_location,
                "Source_x": float(prop["Source_x"][0]), "Source_y": float(prop["Source_y"][0]),
                "Radius": float(methods_cpt["radius"])}

        cpt_BRO = bro.read_bro_gpkg_version(inpt)
        dataframe = cpt_BRO['polygons']['2M81ykd']['data'][4]['dataframe']

        # define target columns and values
        target_columns = ['penetrationLength', 'depth', 'elapsed_time', 'coneResistance',
                          'corrected_cone_resistance',
                          'net_cone_resistance', 'magnetic_field_strength_x', 'magnetic_field_strength_y',
                          'magnetic_field_strength_z',
                          'magnetic_field_strength_total', 'electrical_conductivity', 'inclination_ew',
                          'inclination_ns',
                          'inclination_x', 'inclination_y', 'inclinationResultant',
                          'magnetic_inclination',
                          'magnetic_declination', 'localFriction',
                          'pore_ratio', 'temperature', "porePressureU1", "porePressureU2", "porePressureU3",
                          'frictionRatio', 'id', 'location_x', 'location_y', 'offset_z',
                          'vertical_datum', 'local_reference', 'quality_class',
                          'cpt_standard', 'research_report_date', 'predrilled_z']

        target_values = [0.0, 0.0, 7.0, 0.0, None, None, None, None, None, None, None, None, None, None, None, 0.0,
                         None, None, None, None, None, None, None, None, None, 'CPT000000018803', 82840.1, 443459.9,
                         -0.01, 'NAP', 'maaiveld', 'IMBRO/A', 'NEN5140', '2005-05-20', 0.0]



        # assert if only the columns are read which should be read,
        # assert if the values in the first row of the dataframe are correct
        for idx, column in enumerate(dataframe.columns.values):
            self.assertEqual(column, target_columns[idx])
            # replace all nan values with None in np array
            dataframe = dataframe.replace({np.nan: None})
            if isinstance(target_values[idx], str):
                self.assertTrue(dataframe.values[0][idx], target_values[idx])
            elif target_values[idx] is None:
                self.assertIsNone(dataframe.values[0][idx])
            elif np.isnan(target_values[idx]):
                self.assertTrue(np.isnan(dataframe.values[0][idx]))
            elif isinstance(target_values[idx], (int, float)):
                self.assertAlmostEqual(float(dataframe.values[0][idx]), target_values[idx], places=3)
            else:
                self.assertTrue(False)

    def test_read_cpt(self):
        # inputs
        file_properties = 'unit_testing/unit_testing_files/input_Ground.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 100}
        output = 'unit_testing/unit_testing_files/results/'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)

        # read BRO data base
        inpt = {"BRO_data": prop["BRO_data"],
                "Source_x": float(prop["Source_x"][0]), "Source_y": float(prop["Source_y"][0]),
                "Radius": float(methods_cpt["radius"])}
        cpt_BRO = bro.read_bro_gpkg_version(inpt)

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
        file_properties = 'unit_testing/unit_testing_files/input_Ground.json'
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 100}
        output = 'unit_testing/unit_testing_files/results/'
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
        file_properties = 'unit_testing/unit_testing_files/input_Ground_no_data.json'
        methods_cpt = {'radius': 200}
        output = 'unit_testing/unit_testing_files/results'
        plots = False
        with open(file_properties) as properties:
            prop = json.load(properties)

        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # test if files are filled in correctly
        with open(output + '/' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()
            self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                            '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
            self.assertTrue(logfilelines[1], '# Error # : No data in this coordinate point')
            self.assertTrue(logfilelines[2], '# Info # : Analysis finished for coordinate point: ' +
                            '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')' )

        # test if the second point log is created correctly
        with open(output + '/' + 'log_file_1.txt') as logfile:
            logfilelines = logfile.readlines()
            self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                            '(' + str(prop["Source_x"][1]) + ',' + str(prop["Source_y"][1]) + ')')
            self.assertTrue(logfilelines[1], '# Error # : No data in this coordinate point')
            self.assertTrue(logfilelines[2], '# Info # : Analysis finished for coordinate point: ' +
                            '(' + str(prop["Source_x"][1]) + ',' + str(prop["Source_y"][1]) + ')' )

    def test_analysis_only_circles(self):
        import os
        # inputs
        file_properties = join(dirname(__file__), 'unit_testing_files/input_Ground_only_circle.json')
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 26}
        output = join(dirname(__file__), 'unit_testing_files/results')
        plots = True
        with open(file_properties) as properties:
            prop = json.load(properties)
        # change the bro cpt folder to absolute path
        prop["BRO_data"] = join(dirname(__file__), "..", prop["BRO_data"])
        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # the results cpts
        cpt_results = ['CPT000000002571', 'CPT000000011589']
        # read results
        with open(output + '/' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()

        # check log file
        self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                        '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
        self.assertTrue(logfilelines[-1], '# Info # : Analysis finished for coordinate point: ' +
                        '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
        self.assertTrue(logfilelines[1], '# Info # : Reading CPT: ' + cpt_results[0])
        self.assertTrue(logfilelines[2], '# Info # : Analysis succeeded for: ' + cpt_results[0])
        self.assertTrue(logfilelines[3], '# Info # : Reading CPT: ' + cpt_results[1])
        self.assertTrue(logfilelines[4], '# Info # : Analysis succeeded for: ' + cpt_results[1])

        # check json file
        with open(output + '/' + 'results_0.json') as jsonfile:
            jsonresults = json.load(jsonfile)

        self.assertEqual(jsonresults['scenarios'][0]['probability'], 1.)

        # Check plots
        for i in cpt_results:
            self.assertTrue(os.path.exists(output + '/' + i + '.csv'))
            self.assertTrue(os.path.exists(output + '/' + i + '_cpt.png'))
            self.assertTrue(os.path.exists(output + '/' + i + '_lithology.png'))
        return

    def test_analysis_only_polygon(self):
        import os
        # inputs
        file_properties = join(dirname(__file__), 'unit_testing_files/input_Ground_only_polygon.json')
        methods_cpt = {'gamma': 'Robertson',
                       'vs': 'Robertson',
                       'OCR': 'Mayne',
                       'radius': 26}
        output = join(dirname(__file__), 'unit_testing_files/results')
        plots = True
        with open(file_properties) as properties:
            prop = json.load(properties)
        # change the bro cpt folder to absolute path
        prop["BRO_data"] = join(dirname(__file__), "..", prop["BRO_data"])
        # the function
        cpt_tool.analysis(prop, methods_cpt, self.settings, output, plots)

        # the results cpts
        cpt_results = ['CPT000000002571', 'CPT000000011589']
        # read results
        with open(output + '/' + 'log_file_0.txt') as logfile:
            logfilelines = logfile.readlines()

        # check log file
        self.assertTrue(logfilelines[0], '# Info # : Analysis started for coordinate point: ' +
                        '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
        self.assertTrue(logfilelines[-1], '# Info # : Analysis finished for coordinate point: ' +
                        '(' + str(prop["Source_x"][0]) + ',' + str(prop["Source_y"][0]) + ')')
        self.assertTrue(logfilelines[1], '# Info # : Reading CPT: ' + cpt_results[0])
        self.assertTrue(logfilelines[2], '# Info # : Analysis succeeded for: ' + cpt_results[0])
        self.assertTrue(logfilelines[3], '# Info # : Reading CPT: ' + cpt_results[1])
        self.assertTrue(logfilelines[4], '# Info # : Analysis succeeded for: ' + cpt_results[1])

        # check json file
        with open(output + '/' + 'results_0.json') as jsonfile:
            jsonresults = json.load(jsonfile)

        self.assertEqual(jsonresults['scenarios'][0]['probability'], 1.)

        # Check plots
        for i in cpt_results:
            self.assertTrue(os.path.exists(output + '/' + i + '.csv'))
            self.assertTrue(os.path.exists(output + '/' + i + '_cpt.png'))
            self.assertTrue(os.path.exists(output + '/' + i + '_lithology.png'))
        return

    def tearDown(self):
        import shutil
        import os
        if os.path.exists('unit_testing/unit_testing_files/results'):
            shutil.rmtree('unit_testing/unit_testing_files/results')
        return


class TestGeoMorph(unittest.TestCase):
    """Test reading of the GeoMorph index."""

    def test_index_read(self):
        file_idx = join('bro', 'geomorph')
        gm_index = index.Index(file_idx)  # created by ../shapefiles/gen_geomorph_idx.py

        geomorphs_nl = list(gm_index.intersection(gm_index.bounds, objects="raw"))

        self.assertTrue(isinstance(geomorphs_nl[0][0], type("")))
        self.assertTrue(isinstance(geomorphs_nl[0][1], type({})))
        self.assertEqual(len(geomorphs_nl), 74121)


if __name__ == '__main__':
    unittest.main()

