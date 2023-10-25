import numpy as np
from CPTtool import cpt_tool
import json
import os
import unittest
import shutil
from os.path import join, dirname


class FunctionalTests(unittest.TestCase):
    def setUp(self):

        # reference results
        self.data_ref = [r'./integration_test/results_REF_0.json',
                         r'./integration_test/results_REF_1.json',
                         r'./integration_test/results_REF_2.json']
        # reference results track
        self.data_ref_track = [r'./integration_test/results_track_0.json',
                               r'./integration_test/results_track_1.json']
        # reference results Robertson
        self.data_ref_rob = [r'./integration_test/results_REF_rob_0.json',
                             r'./integration_test/results_REF_rob_1.json',
                             r'./integration_test/results_REF_rob_2.json']

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


    def test_xml(self):
        # test the xml BRO reader
        # run xml
        # set global paths
        file_props = join(dirname(__file__), r'../integration_test/inputs/input_xml.json')
        props = cpt_tool.read_json(file_props)
        file_methods = join(dirname(__file__), r'../integration_test/inputs/methods.json')
        methods = cpt_tool.define_methods(file_methods)
        cpt_tool.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])


    def test_xml_track(self):
        # test the xml BRO reader
        # run xml
        # set global paths
        file_props = join(dirname(__file__), r'../integration_test/inputs/input_xml_track.json')
        props = cpt_tool.read_json(file_props)
        file_methods = join(dirname(__file__), r'../integration_test/inputs/methods.json')
        methods = cpt_tool.define_methods(file_methods)
        methods["radius"] = 1
        cpt_tool.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref_track):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])


    def test_xml_robertson(self):
        # test the xml BRO reader
        # run xml
        props = cpt_tool.read_json(r'./integration_test/inputs/input_xml.json')
        methods = cpt_tool.define_methods(r'./integration_test/inputs/methods_robertson.json')
        cpt_tool.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref_rob):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])


    def assert_dict_almost_equal(self, expected, actual):

        for key in expected:
            if isinstance(expected[key], dict):
                self.assert_dict_almost_equal(expected[key], actual[key])
            else:
                if isinstance(expected[key], (int, float)):
                    self.assertAlmostEqual(expected[key], actual[key])
                elif isinstance(expected[key], list):
                    # check length
                    self.assertEqual(len(expected[key]), len(actual[key]))
                    # if elements are string
                    if all(isinstance(n, str) for n in expected[key]):
                        for i in range(len(expected[key])):
                            self.assertTrue(expected[key][i] == actual[key][i])
                    else:
                        self.assertTrue(all(np.isclose(expected[key], actual[key], rtol=1e-10)))
                elif all(isinstance(n, str) for n in expected[key]):
                    # if elements are string
                    self.assertAlmostEqual(expected[key], actual[key])
                else:
                    self.assertTrue(all(np.isclose(expected[key], actual[key], rtol=1e-10)))

    def tearDown(self):
        # delete folders
        if os.path.exists('./results'):
            shutil.rmtree('./results')


def sort_dicts(dic_ref):
    # sort dics based on coordinates
    # rename the name based on index
    coord = []
    for d in dic_ref:
        coord.append(d["coordinates"][0] + d["coordinates"][1])

    idx = sorted(range(len(coord)), key=lambda k: coord[k])

    ref_ordered = [dic_ref[i] for i in idx]
    for i in range(len(ref_ordered)):
        ref_ordered[i].update({"Name": i})

    return ref_ordered


def read_file(file):
    # reference results
    with open(file, "r") as f:
        data = json.load(f)
        f.close()

    data = sort_dicts(data['scenarios'])

    return data


if __name__ == '__main__':
    unittest.main()
