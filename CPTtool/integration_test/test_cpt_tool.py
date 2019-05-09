import sys
sys.path.append(r'../')
import cpt_tool as cpt
import json
import unittest
import shutil


class FunctionalTests(unittest.TestCase):
    def setUp(self):
        # reference results
        with open(r'./results_REF.json', "r") as f:
            data_ref = json.load(f)

        self.data_ref = sort_dicts(data_ref['scenarios'])
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

    def test_zip(self):
        # test the xml BRO reader
        # run xml
        props = cpt.read_json(r'./inputs/input_zip.json')
        methods = cpt.define_methods(r'./inputs/methods.json')
        cpt.analysis(props, methods, "./results", True)

        # read results
        with open(r'./results/results_0.json', "r") as f:
            data = json.load(f)

        self.assertTrue(self.data_ref == sort_dicts(data['scenarios']))
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


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity

    if is_running_under_teamcity():
        from teamcity.unittestpy import TeamcityTestRunner

        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
