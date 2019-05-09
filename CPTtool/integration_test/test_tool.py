import sys
sys.path.append(r'../')
import cpt_tool as cpt
import json
import unittest
import shutil


class FunctionalTests(unittest.TestCase):
    def setUp(self):

        # reference results
        self.data_ref = read_file(r'./results_REF.json')
        # reference results Robertson
        self.data_ref_rob = read_file(r'./results_REF_rob.json')

        return

    def test_xml(self):
        # test the xml BRO reader
        # run xml
        props = cpt.read_json(r'./inputs/input_xml.json')
        methods = cpt.define_methods(r'./inputs/methods.json')
        cpt.analysis(props, methods, "./results", True)

        # read results
        with open(r'./results/results_0.json', "r") as f:
            data = json.load(f)

        self.assertTrue(self.data_ref == sort_dicts(data['scenarios']))
        return

    def test_xml_robertson(self):
        # test the xml BRO reader
        # run xml
        props = cpt.read_json(r'./inputs/input_xml.json')
        methods = cpt.define_methods(r'./inputs/methods_robertson.json')
        cpt.analysis(props, methods, "./results", True)

        # read results
        with open(r'./results/results_0.json', "r") as f:
            data = json.load(f)

        self.assertTrue(self.data_ref_rob == sort_dicts(data['scenarios']))
        return

    def test_zip(self):
        # test the xml BRO reader
        # run zip
        props = cpt.read_json(r'./inputs/input_zip.json')
        methods = cpt.define_methods(r'./inputs/methods.json')
        cpt.analysis(props, methods, "./results", True)

        # read results
        with open(r'./results/results_0.json', "r") as f:
            data = json.load(f)

        self.assertTrue(self.data_ref == sort_dicts(data['scenarios']))
        return

    def test_zip_robertson(self):
        # test the xml BRO reader
        # run zip
        props = cpt.read_json(r'./inputs/input_zip.json')
        methods = cpt.define_methods(r'./inputs/methods_robertson.json')
        cpt.analysis(props, methods, "./results", True)

        # read results
        with open(r'./results/results_0.json', "r") as f:
            data = json.load(f)

        self.assertTrue(self.data_ref_rob == sort_dicts(data['scenarios']))
        return

    def tearDown(self):
        # delete folders
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
