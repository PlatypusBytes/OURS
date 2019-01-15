# unit test for the cpt_module

import sys
# add the src folder to the path to search for files
sys.path.append('../')
import unittest
import cpt_tool
import numpy as np


class TestCptTool(unittest.TestCase):
    def setUp(self):
        return

    def test_set_key(self):
        local_labels = ['depth', 'tip', 'friction', 'friction_nb', 'water']
        local_dat = [1, 2, 3, 4, 6]
        key = cpt_tool.set_key()
        for i in range(len(local_dat)):
            np.testing.assert_equal(key[local_labels[i]],local_dat[i])
        return

    def tearDown(self):
        return


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity
    if is_running_under_teamcity():
        from teamcity.unittestpy import TeamcityTestRunner
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
