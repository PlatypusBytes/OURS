import sys
sys.path.append(r'../')
import numpy as np
import cpt_tool as cpt
import json
import os
import unittest
import shutil


class FunctionalTests(unittest.TestCase):
    def setUp(self):

        # reference results
        self.data_ref = [r'./results_REF_0.json', r'./results_REF_1.json', r'./results_REF_2.json']
        # reference results Robertson
        self.data_ref_rob = [r'./results_REF_rob_0.json', r'./results_REF_rob_1.json', r'./results_REF_rob_2.json']

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

    def test_xml(self):
        # test the xml BRO reader
        # run xml
        props = cpt.read_json(r'./inputs/input_xml.json')
        methods = cpt.define_methods(r'./inputs/methods.json')
        cpt.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)
                f.close()

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])

        return

    def test_xml_robertson(self):
        # test the xml BRO reader
        # run xml
        props = cpt.read_json(r'./inputs/input_xml.json')
        methods = cpt.define_methods(r'./inputs/methods_robertson.json')
        cpt.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref_rob):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)
                f.close()

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])
        return

    def test_zip(self):
        # test the xml BRO reader
        # run zip
        props = cpt.read_json(r'./inputs/input_zip.json')
        methods = cpt.define_methods(r'./inputs/methods.json')
        cpt.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)
                f.close()

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])

        return

    def test_zip_robertson(self):
        # test the xml BRO reader
        # run zip
        props = cpt.read_json(r'./inputs/input_zip.json')
        methods = cpt.define_methods(r'./inputs/methods_robertson.json')
        cpt.analysis(props, methods, self.settings, "./results", False)

        # for the points of analysis
        for i, fil in enumerate(self.data_ref_rob):
            data_ref = read_file(fil)

            # read results
            with open(r'./results/results_' + str(i) + '.json', 'r') as f:
                data = json.load(f)
                f.close()

            # compare dics
            for k in range(len(data_ref)):
                self.assert_dict_almost_equal(data_ref[k], sort_dicts(data['scenarios'])[k])

        return

    def assert_dict_almost_equal(self, expected, actual):

        for key in expected:
            if isinstance(expected[key], dict):
                self.assert_dict_almost_equal(expected[key], actual[key])
            else:
                if isinstance(expected[key], (int, float)):
                    self.assertAlmostEqual(expected[key], actual[key])
                elif all(isinstance(n, str) for n in expected[key]):
                    # if elements are string
                    self.assertAlmostEqual(expected[key], actual[key])
                else:
                    self.assertTrue(all(np.isclose(expected[key], actual[key], rtol=1e-10)))
        return

    def tearDown(self):
        # delete folders
        if os.path.exists('./results'):
            shutil.rmtree('./results')
        return


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


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity

    if is_running_under_teamcity():
        from teamcity.unittestpy import TeamcityTestRunner

        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)