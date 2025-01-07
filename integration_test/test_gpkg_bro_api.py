from os.path import join, dirname
import unittest
import pickle
import pandas as pd
import numpy as np

from CPTtool import bro

class TestBroDb(unittest.TestCase):
    """Test creation of index and parsing of database."""

    def test_database_read_circle(self):
        # get absolute path to the test database
        test_db = join(dirname(__file__), '../bro/test_v2_0_1.gpkg')
        input = {"BRO_data": test_db, "Source_x": 82900, "Source_y": 443351, "Radius": 1}
        cpts = bro.read_bro_gpkg_version(input)

        # data bro_data.pickle was created with
        ## from BroReader import read_BRO
        ## c = read_BRO.read_cpts([input["Source_x"], input["Source_y"]], input["Radius"] * 0.001, interpret_cpt=False)
        ## c[0]['plot_settings'] = None
        ## with open("data_bro.pickle", "wb") as f:
        ##     pickle.dump(c, f)

        with open("./integration_test/inputs/data_bro.pickle", "rb") as f:
            c = pickle.load(f)

        # compare the number of cpts read from the database and the number of cpts read from the file
        self.assertEqual(len(cpts["circle"]["data"]), len(c))
        target_cpt = cpts['circle']['data'][0]
        self.assertEqual(target_cpt['id'], c[0]['name'])
        self.assertEqual(target_cpt['location_x'], c[0]['coordinates'][0])
        self.assertEqual(target_cpt['location_y'], c[0]['coordinates'][1])
        # check that arrays are the same
        list_to_check = [ ["depth", "depth"],
                          ['penetrationLength', 'penetration_length'],
                          ['coneResistance', 'tip'],
                          ["localFriction", "friction"]]
        target_cpt['dataframe'] = target_cpt['dataframe'].where(pd.notnull(target_cpt['dataframe']), np.nan)
        for i in range(len(list_to_check)):
            # check arrays with almost equal
            target = target_cpt['dataframe'][list_to_check[i][0]].to_numpy()
            actual = c[0][list_to_check[i][1]]
            self.assertTrue((np.isclose(target, actual) | np.isnan(target) |  np.isnan(target)).all())
