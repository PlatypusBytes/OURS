# unit test for the cpt_module
import sys
# add the src folder to the path to search for files
sys.path.append('../')
import unittest
import cpt_module
import numpy as np
import cpt_tool
import pandas as pd
import csv
import os


class TestCptModule(unittest.TestCase):
    def setUp(self):
        import cpt_module
        # Initiating the cpt_module
        self.cpt = cpt_module.CPT("./")
        pass

    def test__pre_drill_with_predrill(self):

        # make a cpt with the pre_drill option
        d = {'penetrationLength': [1.5, 2.0, 2.5],
             'coneResistance': [1, 2, 3],
             'localFriction': [4, 5, 6],
             'frictionRatio': [0.22, 0.33, 0.44],
             }

        # set up the upper part of the dictionary
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 1.5,
                    'a': 0.8,
                    "dataframe": df}

        # Run the function to be checked
        self.cpt.parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-given lists
        np.testing.assert_array_equal(self.cpt.tip, [1000, 1000, 1000, 1000, 2000, 3000])
        np.testing.assert_array_equal(self.cpt.friction, [4000, 4000, 4000, 4000, 5000, 6000])
        np.testing.assert_array_equal(self.cpt.friction_nbr, [0.22, 0.22, 0.22, 0.22, 0.33, 0.44])
        np.testing.assert_array_equal(self.cpt.depth, [0, 0.5, 1, 1.5, 2, 2.5])
        np.testing.assert_array_equal(self.cpt.NAP, [cpt_data["offset_z"] - i for i in [0, 0.5, 1, 1.5, 2, 2.5]])
        np.testing.assert_array_equal(self.cpt.water, [0., 0., 0., 0., 0., 0.])
        np.testing.assert_array_equal(self.cpt.coord, [cpt_data["location_x"], cpt_data["location_y"]])
        np.testing.assert_equal(self.cpt.name, "cpt_name")
        np.testing.assert_equal(self.cpt.a, 0.8)

        return

    def test__pre_drill_with_pore_pressure(self):

        # Set the values of the cpt
        d = {'penetrationLength': [1.5, 2.0, 2.5],
             'coneResistance': [1, 2, 3],
             'localFriction': [4, 5, 6],
             'frictionRatio': [0.22, 0.33, 0.44],
             'porePressureU2': [1, 2, 3]
             }
        df = pd.DataFrame(data=d)

        # Build the upper part of the library
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 1.5,
                    'a': 0.73,
                    "dataframe": df}

        # define the pore pressure array before the predrilling
        # Here 3 values as the stepping is defined that way.
        # Define the stepping of the pore pressure
        # Then my target values
        # Finally multiply with 1000
        step = 1/len(d['penetrationLength'])
        pore_pressure = [0, step * 1000, 2 * step * 1000, 1 * 1000, 2 * 1000, 3 * 1000]

        # run the function to be checked
        self.cpt.parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-defined values
        np.testing.assert_array_equal(self.cpt.water, pore_pressure)
        np.testing.assert_array_equal(self.cpt.tip, [1000, 1000, 1000, 1000, 2000, 3000])
        np.testing.assert_array_equal(self.cpt.friction, [4000, 4000, 4000, 4000, 5000, 6000])
        np.testing.assert_array_equal(self.cpt.friction_nbr, [0.22, 0.22, 0.22, 0.22, 0.33, 0.44])
        np.testing.assert_array_equal(self.cpt.depth, [0, 0.5, 1, 1.5, 2, 2.5])
        np.testing.assert_array_equal(self.cpt.NAP, [cpt_data["offset_z"] - i for i in [0, 0.5, 1, 1.5, 2, 2.5]])
        np.testing.assert_array_equal(self.cpt.coord, [cpt_data["location_x"], cpt_data["location_y"]])
        np.testing.assert_equal(self.cpt.name, "cpt_name")
        np.testing.assert_equal(self.cpt.a, 0.73)
        return

    def test__pre_drill_Raise_Exception1(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {'penetrationLength': [1.5, 2.0],
             'coneResistance': [1, 2],
             'localFriction': [4, 5],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 1.5,
                    'a': 0.73,
                    "dataframe": df}

        # run the fuction
        aux = self.cpt.parse_bro(cpt_data, minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        self.assertTrue('File cpt_name has a length smaller than 10' == aux)
        return

    def test__pre_drill_Raise_Exception2(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {'penetrationLength': [1.5, 2.0],
             'coneResistance': [1, 2],
             'localFriction': [4, 5],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 1.5,
                    'a': 0.6,
                    "dataframe": df}

        # run the fuction
        aux = self.cpt.parse_bro(cpt_data, minimum_length=1, minimum_samples=10)

        # check if the returned message is the appropriate
        self.assertTrue( 'File cpt_name has a number of samples smaller than 10'== aux)
        return

    def test_read_BRO_Raise_Exception1(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {'penetrationLength': [1.5, 20.0],
             'coneResistance': [-1, 2],
             'localFriction': [4, 5],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 1.5,
                    'a': 0.5,
                    "dataframe": df}

        # run the fuction
        aux = self.cpt.parse_bro(cpt_data, minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        self.assertTrue('File cpt_name is corrupted' == aux)
        return

    def test_read_BRO_Raise_Exception2(self):

        # Define the cpt values
        # here the points are only two so that will return an error message
        d = {'penetrationLength': [1.5, 20.0],
             'coneResistance': [1, 2],
             'localFriction': [-4, 5],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 1.5,
                    'a': 0.5,
                    "dataframe": df}

        # run the fuction
        aux = self.cpt.parse_bro(cpt_data, minimum_length=10, minimum_samples=1)

        # check if the returned message is the appropriate
        self.assertTrue('File cpt_name is corrupted' == aux)
        return

    def test_rho_calculation(self):
        self.cpt.gamma = np.ones(10)
        self.cpt.g = 10.
        self.cpt.rho_calc()

        # exact solution = gamma / g
        exact_rho = np.ones(10) * 1000 / 10

        # self.assertEqual(exact_rho, self.cpt.rho)
        np.testing.assert_array_equal(exact_rho, self.cpt.rho)
        return

    def test_gamma_calc(self):
        # Set all the values
        gamma_limit = 22
        self.cpt.friction_nbr = np.ones(10)
        self.cpt.qt = np.ones(10)
        self.cpt.Pa = 100
        self.cpt.depth = range(10)
        self.cpt.name = 'UNIT_TEST'

        # Calculate analytically the solution
        np.seterr(divide="ignore")
        # Exact solution Robertson
        aux = 0.27 * np.log10(np.ones(10)) + 0.36 * (np.log10(np.ones(10) / 100)) + 1.236
        aux[np.abs(aux) == np.inf] = gamma_limit / 9.81
        local_gamma1 = aux * 9.81

        # call the function to be checked
        self.cpt.gamma_calc()

        # Check if they are equal
        np.testing.assert_array_equal(local_gamma1, self.cpt.gamma)

        # Exact solution Lengkeek
        local_gamma2 = 19 - 4.12 * ((np.log10(5000 / self.cpt.qt)) / (np.log10(30 / self.cpt.friction_nbr)))
        self.cpt.gamma_calc(gamma_max=gamma_limit, method='Lengkeek')
        np.testing.assert_array_equal(local_gamma2, self.cpt.gamma)

        # all of them
        self.cpt.gamma_calc(gamma_max=gamma_limit, method='all')

        import os.path
        self.assertTrue(os.path.isfile('UNIT_TEST_unit_weight.png'))
        return

    def test_stress_calc(self):
        # Defining the inputs of the function
        self.cpt.depth = np.arange(0, 2, 0.1)
        self.cpt.gamma = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        self.cpt.NAP = np.zeros(20)
        self.pwp = 0
        self.cpt.stress_calc()

        # The target list with the desired output
        effective_stress_test = [2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 21.5, 23., 24.5, 26., 27.5, 29., 30.5,
                                 32., 33.5, 35.]

        # checking equality with the output
        np.testing.assert_array_equal(effective_stress_test, list(np.around(self.cpt.effective_stress, 1)))
        return

    def test_norm_calc(self):
        # Empty list that will be filled by reading the csv file
        test_Qtn, total_stress, effective_stress, Pa, tip, friction = [], [], [], [], [], []
        test_Fr = []

        # Opening and reading the csv file
        # These are also the inputs of the function
        with open('unit_testing_files/test_norm_calc.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    total_stress.append(float(row[0]))
                    effective_stress.append(float(row[1]))
                    Pa.append(float(row[2]))
                    tip.append(float(row[3]))
                    friction.append(float(row[4]))

                    # These will be the outputs of the function
                    test_Qtn.append(float(row[5]))
                    test_Fr.append(float(row[6]))
                line_count += 1

        # Allocate them to the cpt
        self.cpt.total_stress = np.array(total_stress)
        self.cpt.effective_stress = np.array(effective_stress)
        self.cpt.Pa = np.array(Pa)
        self.cpt.tip = np.array(tip)
        self.cpt.friction = np.array(friction)
        self.cpt.friction_nbr = np.array(friction)
        self.cpt.norm_calc(n_method=True)

        # Test the equality of the arrays
        np.testing.assert_array_equal(test_Qtn, self.cpt.Qtn)
        np.testing.assert_array_equal(test_Fr, self.cpt.Fr)
        return

    def test_IC_calc(self):
        # Set the inputs of the values
        test_IC = [3.697093]
        self.cpt.Qtn = [1]
        self.cpt.Fr = [1]
        self.cpt.IC_calc()

        # Check if they are equal with the target value test_IC
        np.testing.assert_array_equal(list(np.around(np.array(test_IC), 1)), list(np.around(self.cpt.IC, 1)))
        return

    def test_vs_calc(self):
        # Define all the inputs
        self.cpt.IC = np.array([1])
        self.cpt.Qtn = np.array([1])
        self.cpt.rho = np.array([1])
        self.cpt.total_stress = np.array([1])
        self.cpt.effective_stress = np.array([1])
        self.cpt.tip = np.array([2])
        self.cpt.qt = np.array([2])
        self.Pa = 100
        self.cpt.gamma = np.array([10])
        self.cpt.vs = np.array([1])
        self.cpt.depth = np.array([1])
        self.cpt.Fr = np.array([1])
        self.cpt.name = "UNIT_TESTING"

        # Check the results for Robertson
        # Calculate analytically
        test_alpha_vs = 10 ** (0.55 * self.cpt.IC + 1.68)
        test_vs = (test_alpha_vs * (self.cpt.tip - self.cpt.total_stress) / 100) ** 0.5
        test_GO = self.cpt.rho * test_vs ** 2

        # Call function
        self.cpt.vs_calc(method="Robertson")

        # Check their equality
        np.testing.assert_array_equal(test_vs, self.cpt.vs)
        np.testing.assert_array_equal(test_GO, self.cpt.G0)

        # Check the results for  Mayne
        # Calculate analytically
        test_vs = np.exp((self.cpt.gamma + 4.03) / 4.17) * (self.cpt.effective_stress / self.Pa) ** 0.25
        test_GO = self.cpt.rho * test_vs ** 2

        # Call function
        self.cpt.vs_calc(method="Mayne")

        # Check their equality
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)

        # Check the results for Andrus
        # Calculate analytically
        test_vs = 2.27 * self.cpt.qt ** 0.412 * self.cpt.IC ** 0.989 * self.cpt.depth ** 0.033 * 1
        test_GO = self.cpt.rho * test_vs ** 2

        # Call function
        self.cpt.vs_calc(method="Andrus")

        # Check their equality
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)

        # Check the results for Zang
        # Calculate analytically
        test_vs = 10.915 * self.cpt.tip ** 0.317 * self.cpt.IC ** 0.21 * self.cpt.depth ** 0.057 * 0.92
        test_GO = self.cpt.rho * test_vs ** 2

        # Call function
        self.cpt.vs_calc(method="Zang")

        # Check their equality
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)

        # Check the results for Ahmed
        # Calculate analytically
        test_vs = 1000 * np.e ** (-0.887 * self.cpt.IC) * (
                    1 + 0.443 * self.cpt.Fr * self.cpt.effective_stress / 100 * 9.81 / self.cpt.gamma) ** 0.5
        test_GO = self.cpt.rho * test_vs ** 2

        # Call the function
        self.cpt.vs_calc(method="Ahmed")

        # Check their equality
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)

        # Check for All
        # Call the function
        self.cpt.vs_calc(method="all")

        # Check if the files are outputted in the directory
        self.assertTrue(os.path.isfile("UNIT_TESTING_shear_modulus.png"))
        self.assertTrue(os.path.isfile("UNIT_TESTING_shear_wave.png"))
        return

    def test_poisson_calc(self):
        # Set the inputs
        self.cpt.lithology = ['1', '2', '3', '4', '5', "6", "7", "8", "9"]

        # Set the target outputs
        test_poisson = [0.495, 0.495, 0.495, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3]

        # Call the function
        self.cpt.poisson_calc()

        # Check if they are equal
        np.testing.assert_array_equal(test_poisson, self.cpt.poisson)
        return

    def test_damp_calc_1(self):
        # all soil sensitive : damping = minimum value
        self.cpt.lithology = ["1", "1", "1"]
        self.cpt.effective_stress = np.ones(len(self.cpt.lithology))
        self.cpt.total_stress = np.ones(len(self.cpt.lithology)) + 1
        self.cpt.qt = np.ones(len(self.cpt.lithology)) * 10

        # Define the target array
        test_damping =  2.512 * (self.cpt.effective_stress / 100) ** -0.2889
        test_damping /= 100

        # Running the function
        self.cpt.damp_calc()

        # Testing if the lists are equals
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return

    def test_damp_calc_2(self):
        # Defining the inputs
        # all soil very stiff : damping = minimum value
        self.cpt.lithology = ["8", "8", "8"]
        self.cpt.effective_stress = np.ones(len(self.cpt.lithology))
        self.cpt.total_stress = np.ones(len(self.cpt.lithology)) + 1
        self.cpt.qt = np.ones(len(self.cpt.lithology)) * 10

        # The target output
        Cu = 2
        D50 = 0.02
        test_damping = 0.55 * Cu ** 0.1 * D50 ** -0.3 * (self.cpt.effective_stress / 100) ** -0.08
        test_damping /= 100

        # Run the function to be tested
        self.cpt.damp_calc(Cu=Cu, D50=D50)

        # Testing if the lists are equals
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return

    def test_damp_calc_3(self):
        # all soil grained : damping = minimum value
        # Setting for testing
        self.cpt.lithology = ["9", "9", "9"]
        self.cpt.effective_stress = np.ones(len(self.cpt.lithology))
        self.cpt.total_stress = np.ones(len(self.cpt.lithology)) + 1
        self.cpt.qt = np.ones(len(self.cpt.lithology)) * 10

        # Define the output
        test_damping = np.array([2, 2, 2]) / 100

        Cu = 3
        D50 = 0.025
        test_damping = 0.55 * Cu ** 0.1 * D50 ** -0.3 * (self.cpt.effective_stress / 100) ** -0.08
        test_damping /= 100

        # Run the function to be tested
        self.cpt.damp_calc(Cu=Cu, D50=D50)

        # Testing if the list are equal
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return

    def test_damp_calc_4(self):
        # Define the inputs
        # all soil sand
        self.cpt.lithology = ["8", "6", "9", "7"]
        self.cpt.effective_stress = np.ones(len(self.cpt.lithology))
        self.cpt.total_stress = np.ones(len(self.cpt.lithology)) + 1
        self.cpt.qt = np.ones(len(self.cpt.lithology)) * 10

        # Calculate analytically for the type Meng
        Cu = 3
        D50 = .025
        test_damping = 0.55 * Cu ** 0.1 * D50 ** -0.3 * (self.cpt.effective_stress / 100) ** -0.08
        test_damping /= 100

        # Run the function to be tested
        self.cpt.damp_calc(Cu=Cu, D50=D50)

        # Testing if the list are equal
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return

    def test_damp_calc_5(self):
        # Define the inputs
        # all soil sand
        self.cpt.lithology = ["3", "4", "3", "4"]
        self.cpt.qt = np.ones(len(self.cpt.lithology)) * 10
        self.cpt.Qtn = np.ones(len(self.cpt.lithology)) * 2
        self.cpt.effective_stress = np.ones(len(self.cpt.lithology))
        self.cpt.total_stress = np.ones(len(self.cpt.lithology)) + 1

        # Calculate analyticaly damping Darendeli - OCR according to Mayne
        Cu = 3
        D50 = .025
        PI = 40
        OCR = 0.33 * (self.cpt.qt - self.cpt.total_stress) / self.cpt.effective_stress
        freq = 1
        test_damping = (self.cpt.effective_stress / 100) ** (-0.2889) * \
                       (0.8005 + 0.0129 * PI * OCR ** (-0.1069)) * (1 + 0.2919 * np.log(freq))
        test_damping /= 100

        # Call the function to be tested
        self.cpt.damp_calc(Cu=Cu, D50=D50, Ip=PI, method="Mayne")

        # Test the damping Darendeli - OCR according to Mayne
        np.testing.assert_array_equal(test_damping, self.cpt.damping)

        # Calculated analyticaly damping Darendeli - OCR according to robertson
        OCR = 0.25 * (self.cpt.Qtn) ** 1.25
        test_damping = (self.cpt.effective_stress / 100) ** (-0.2889) * \
                       (0.8005 + 0.0129 * PI * OCR ** (-0.1069)) * (1 + 0.2919 * np.log(freq))
        test_damping /= 100

        self.cpt.damp_calc(Cu=Cu, D50=D50, Ip=PI, method="Robertson")
        return

    def test_damp_calc_6(self):
        # Set the inputs
        # all soil sand
        self.cpt.lithology = ["2", "2", "2", "2"]
        self.cpt.effective_stress = np.ones(len(self.cpt.lithology))
        self.cpt.Pa = 1

        # Calculate analyticaly
        test_damping = (2.512/100)*np.ones(len(self.cpt.lithology))

        # Call the function to be tested
        self.cpt.damp_calc()

        # Check if they are equal
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return

    def test_damp_calc_7(self):
        # stress is zero so damping is infinite
        self.cpt.lithology = ["1", "1", "1"]
        self.cpt.effective_stress = np.zeros(len(self.cpt.lithology))
        self.cpt.total_stress = np.zeros(len(self.cpt.lithology)) + 1
        self.cpt.qt = np.ones(len(self.cpt.lithology)) * 10

        # Define the target array
        test_damping = [1, 1, 1]

        # Running the function
        self.cpt.damp_calc()

        # Testing if the lists are equals
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return

    def test_qt_calc(self):
        # Define the inputs
        self.cpt.tip = np.array([1])
        self.cpt.water = np.array([1])
        self.cpt.a = np.array([1])

        # Define the target
        test_qt = np.array([1])

        # Call the function to be tested
        self.cpt.qt_calc()

        # Check if the are equal
        np.testing.assert_array_equal(test_qt, self.cpt.qt)
        return

    def test_write_csv(self):
        # Create the file
        self.cpt.name = 'UNIT_TEST'
        self.cpt.write_csv()

        # Check if the file exist in the directory
        self.assertTrue(os.path.isfile('UNIT_TEST.csv'))
        return

    def test_plot_lithology(self):
        # Create the file
        self.cpt.name = 'UNIT_TEST'
        self.cpt.plot_lithology()

        # Check if the file exist in the directory
        self.assertTrue(os.path.isfile('UNIT_TEST_lithology.png'))
        return

    def test_plot_cpt(self):
        # Create the file
        self.cpt.name = 'UNIT_TEST'
        self.cpt.plot_cpt()

        # Check if the file exist in the directory
        self.assertTrue(os.path.isfile('UNIT_TEST_cpt.png'))
        return

    def test_plot_correlations(self):
        # Create the file
        self.cpt.name = 'UNIT_TEST'
        self.cpt.plot_correlations([], 'Correlations', 'Correlations', 'Correlations')

        # Check if the file exist in the directory
        self.assertTrue(os.path.isfile('UNIT_TEST_Correlations.png'))
        return

    def test_lithology_calc(self):
        # Define the input
        self.cpt.tip = np.array([1])
        self.cpt.friction_nbr = np.array([1])
        self.cpt.friction = np.array([1])
        self.cpt.effective_stress = np.array([2])
        self.cpt.total_stress = np.array([2])
        self.cpt.Pa = 100
        lithology_test = ['1']

        # Call the function to be tested
        self.cpt.lithology_calc()

        # Check if results are equal
        self.assertEqual(self.cpt.lithology, lithology_test)
        return

    def test_bro_parser_no_water(self):

        d = {'penetrationLength': [0.1, 0.2],
             'coneResistance': [1, 2],
             'localFriction': [3, 4],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    'predrilled_z': 0.,
                    'a': 0.85,
                    "dataframe": df}

        self.cpt.parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-given lists
        np.testing.assert_array_equal(self.cpt.tip, [1000, 2000])
        np.testing.assert_array_equal(self.cpt.friction, [3000, 4000])
        np.testing.assert_array_equal(self.cpt.friction_nbr, [0.22, 0.33])
        np.testing.assert_array_equal(self.cpt.depth, [0, 0.1])
        np.testing.assert_array_equal(self.cpt.NAP, [cpt_data["offset_z"] - i for i in [0, 0.1]])
        np.testing.assert_array_equal(self.cpt.water, [0., 0.])
        np.testing.assert_array_equal(self.cpt.coord, [cpt_data["location_x"], cpt_data["location_y"]])
        np.testing.assert_equal(self.cpt.name, "cpt_name")
        np.testing.assert_equal(self.cpt.a, 0.85)

        return

    def test_bro_parser_water(self):

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
        self.cpt.parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-given lists
        np.testing.assert_array_equal(self.cpt.tip, [1000, 2000])
        np.testing.assert_array_equal(self.cpt.friction, [3000, 4000])
        np.testing.assert_array_equal(self.cpt.friction_nbr, [0.22, 0.33])
        np.testing.assert_array_equal(self.cpt.depth, [0, 0.1])
        np.testing.assert_array_equal(self.cpt.NAP, [cpt_data["offset_z"] - i for i in [0, 0.1]])
        np.testing.assert_array_equal(self.cpt.water, [500., 1000.])
        np.testing.assert_array_equal(self.cpt.coord, [cpt_data["location_x"], cpt_data["location_y"]])
        np.testing.assert_equal(self.cpt.name, "cpt_name")
        np.testing.assert_equal(self.cpt.a, 0.85)
        return

    def test_bro_parser_nan(self):

        d = {'penetrationLength': [0.1, 0.2],
             'coneResistance': [1, 2],
             'localFriction': [3, np.nan],
             'porePressureU2': [0.5, 1],
             'frictionRatio': [0.22, 0.33],
             }
        df = pd.DataFrame(data=d)
        cpt_data = {"id": "cpt_name",
                    "location_x": 111,
                    "location_y": 222,
                    "offset_z": 0.5,
                    "predrilled_z": 0.,
                    "a": 0.8,
                    "dataframe": df}

        self.cpt.parse_bro(cpt_data, minimum_length=0.01, minimum_samples=1)

        # Check the equality with the pre-given lists
        np.testing.assert_array_equal(self.cpt.tip, [1000])
        np.testing.assert_array_equal(self.cpt.friction, [3000])
        np.testing.assert_array_equal(self.cpt.friction_nbr, [0.22])
        np.testing.assert_array_equal(self.cpt.depth, [0])
        np.testing.assert_array_equal(self.cpt.NAP, [cpt_data["offset_z"] - i for i in [0]])
        np.testing.assert_array_equal(self.cpt.water, [500.])
        np.testing.assert_array_equal(self.cpt.coord, [cpt_data["location_x"], cpt_data["location_y"]])
        np.testing.assert_equal(self.cpt.name, "cpt_name")
        np.testing.assert_equal(self.cpt.a, 0.8)
        return

    def tearDown(self):
        # Delete all files created while testing
        import os
        list_delete = ["UNIT_TEST.csv", "UNIT_TEST_Correlations.png", "UNIT_TEST_cpt.png", "UNIT_TEST_lithology.png",
                       "UNIT_TEST_unit_weight.png", "UNIT_TESTING_shear_modulus.png", "UNIT_TESTING_shear_wave.png"]
        for i in list_delete:
            if os.path.exists(i):
                os.remove(i)
        return


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity

    if is_running_under_teamcity():
        from teamcity.unittestpy import TeamcityTestRunner

        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
