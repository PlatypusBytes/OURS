# unit test for the cpt_module

import sys
import numpy as np
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner

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


class TestBroDb(unittest.TestCase):
    """Test creation of index and parsing of database."""

    def test_database_read(self):
        input = {"BRO_data": "../bro/brocpt.xml", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro(input)

        self.assertEqual(len(cpts), 23)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts))), 2)

    def test_zipdatabase_read(self):
        input = {"BRO_data": "../bro/brocpt.zip", "Source_x": 82900, "Source_y": 443351, "Radius": 100}
        cpts = bro.read_bro(input)

        self.assertEqual(len(cpts), 23)
        self.assertEqual(len(list(filter(lambda x: x is None, cpts))), 2)


if __name__ == '__main__':  # pragma: no cover
    if is_running_under_teamcity():
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
