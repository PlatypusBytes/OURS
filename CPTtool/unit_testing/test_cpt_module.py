# unit test for the cpt_module

import sys
# add the src folder to the path to search for files
sys.path.append('../')
import unittest
import cpt_module
import numpy as np
import cpt_tool

class TestCptModule(unittest.TestCase):
    def setUp(self):
        import cpt_module
        import log_handler
        self.log_file = log_handler.LogFile("./results")

        self.cpt = cpt_module.CPT("./", self.log_file)
        pass

    def test_read_gef(self):
        gef_file = 'unit_testing.gef'
        key_cpt = cpt_tool.set_key()
        self.cpt.read_gef(gef_file, key_cpt)
        test_name = 'UNIT_TESTING'
        test_coord = [244319.00, 587520.00]

        test_depth = range(20)
        test_NAP = test_depth*np.full(20,-1) + np.full(20,-0.87)
        test_tip = np.full(20,1000)
        test_friction = np.full(20,2000)
        test_friction_nbr = np.full(20,5)
        test_water = np.full(20,3000)

        np.testing.assert_array_equal(test_name, self.cpt.name)
        np.testing.assert_array_equal(test_coord, self.cpt.coord)
        np.testing.assert_array_equal(test_depth, self.cpt.depth)
        np.testing.assert_array_equal(test_NAP, self.cpt.NAP)
        np.testing.assert_array_equal(test_tip, self.cpt.tip)
        np.testing.assert_array_equal(test_friction, self.cpt.friction)
        np.testing.assert_array_equal(test_friction_nbr, self.cpt.friction_nbr)
        np.testing.assert_array_equal(test_water, self.cpt.water)

        # Exceptions tested
        gef_file = 'Exception_NoNAP.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_NoCoord.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_NoLength.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_NoTip.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_NoFriction.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_NoFrictionNumber.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_NoWater.gef'
        self.assertFalse(self.cpt.read_gef(gef_file, key_cpt))
        gef_file = 'Exception_9999.gef'
        self.assertTrue(self.cpt.read_gef(gef_file, key_cpt))

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
        gamma_limit = 22
        self.cpt.friction_nbr = np.ones(10)
        self.cpt.qt = np.ones(10)
        self.cpt.Pa = 100
        self.cpt.depth = range(10)
        self.cpt.name = 'UNIT_TEST'
        np.seterr(divide="ignore")
        # Exact solution Robertson
        aux = 0.27*np.log10(np.ones(10))+0.36*(np.log10(np.ones(10)/ 100))+1.236
        aux[np.abs(aux) == np.inf] = gamma_limit / 9.81
        local_gamma1 = aux * 9.81
        self.cpt.gamma_calc(gamma_limit)
        np.testing.assert_array_equal(local_gamma1, self.cpt.gamma)

        # Exact solution Lengkeek
        local_gamma2 = 19 - 4.12*((np.log10(5000/self.cpt.qt))/(np.log10(30/self.cpt.friction_nbr)))
        self.cpt.gamma_calc(gamma_limit,method='Lengkeek')
        np.testing.assert_array_equal(local_gamma2, self.cpt.gamma)

        #all of them
        self.cpt.gamma_calc(gamma_limit, method='all')

        import os.path
        self.assertTrue(os.path.isfile('UNIT_TEST_unit_weight.png'))
        return

    def test_merge_thickness(self):
        min_layer_thick = 0.5
        self.cpt.depth = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        self.cpt.lithology = ['0','0','0','0','0','1','2','2','2','2','2','2']
        self.cpt.IC = [0.5,0.5,0.5,0.5,0.5,1,3,3,3,3,3,3]
        merged = self.cpt.merge_thickness(min_layer_thick)
        depth_test = [0.0,0.6,1.1]
        test_lithology = ['/0/1','/2']
        test_index = [0,6]
        np.testing.assert_array_equal(depth_test, self.cpt.depth_json)
        np.testing.assert_array_equal(test_lithology, self.cpt.lithology_json)
        np.testing.assert_array_equal(test_index, self.cpt.indx_json)

        self.cpt.IC = [3,3,3,3,3,1,0.5,0.5,0.5,0.5,0.5,0.5]
        merged = self.cpt.merge_thickness(min_layer_thick)
        depth_test = [0.0,0.5,1.1]
        test_lithology = ['/0','/1/2']
        test_index = [0,5]
        np.testing.assert_array_equal(depth_test, self.cpt.depth_json)
        np.testing.assert_array_equal(test_lithology, self.cpt.lithology_json)
        np.testing.assert_array_equal(test_index, self.cpt.indx_json)
        return


    def test_stress_calc(self):
         self.cpt.depth = np.arange(0,2,0.1)
         self.cpt.gamma = [20,20,20,20,20,20,20,20,20,20,15,15,15,15,15,15,15,15,15,15]
         self.cpt.NAP = np.zeros(20)
         z_pwp = 0
         self.cpt.stress_calc(z_pwp)
         effective_stress_test = [2.,4.,6.,8.,10.,12.,14.,16.,18.,20.,21.5,23.,24.5,26.,27.5,29.,30.5,32.,33.5,35.]

         np.testing.assert_array_equal(effective_stress_test,list(np.around(self.cpt.effective_stress,1)) )
         return

    def test_norm_calc(self):
         import csv
         test_Qtn ,total_stress, effective_stress ,Pa , tip ,friction = [] ,[],[],[],[],[]
         test_Fr = []

         with open('test_norm_calc.csv') as csv_file:
             csv_reader = csv.reader(csv_file, delimiter=';')
             line_count = 0
             for row in csv_reader:
                 if line_count != 0:
                     total_stress.append(float(row[0]))
                     effective_stress.append(float(row[1]))
                     Pa.append(float(row[2]))
                     tip.append(float(row[3]))
                     friction.append(float(row[4]))
                     test_Qtn.append(float(row[5]))
                     test_Fr.append(float(row[6]))
                 line_count += 1
         self.cpt.total_stress = np.array(total_stress)
         self.cpt.effective_stress = np.array(effective_stress)
         self.cpt.Pa = np.array(Pa)
         self.cpt.tip = np.array(tip)
         self.cpt.friction = np.array(friction)
         self.cpt.friction_nbr = np.array(friction)
         self.cpt.norm_calc(n_method=True)
         np.testing.assert_array_equal(test_Qtn,self.cpt.Qtn)
         np.testing.assert_array_equal(test_Fr, self.cpt.Fr)

         self.cpt.norm_calc(n_method=False)
         np.testing.assert_array_equal(test_Qtn,self.cpt.Qtn)
         np.testing.assert_array_equal(test_Fr, self.cpt.Fr)

         return

    def test_IC_calc(self):
        test_IC = [3.697093]
        self.cpt.Qtn = [1]
        self.cpt.Fr = [1]
        self.cpt.IC_calc()
        np.testing.assert_array_equal(list(np.around(np.array(test_IC),1)), list(np.around(self.cpt.IC,1)))
        return

    def test_vs_calc(self):
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
        # Robertson
        test_alpha_vs = 10**(0.55*self.cpt.IC+1.68)
        test_vs = (test_alpha_vs*(self.cpt.tip - self.cpt.total_stress)/100)**0.5
        test_GO = self.cpt.rho * test_vs**2
        self.cpt.vs_calc(method="Robertson")
        np.testing.assert_array_equal(test_vs, self.cpt.vs)
        np.testing.assert_array_equal(test_GO, self.cpt.G0)
        # Mayne
        test_vs = np.exp((self.cpt.gamma+4.03)/4.17)*(self.cpt.effective_stress/self.Pa)**0.25
        test_GO = self.cpt.rho * test_vs ** 2
        self.cpt.vs_calc(method="Mayne")
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)
        # Andrus
        test_vs = 2.27 * self.cpt.qt**0.412 * self.cpt.IC**0.989 * self.cpt.depth **0.033*1
        test_GO = self.cpt.rho * test_vs ** 2
        self.cpt.vs_calc(method="Andrus")
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)
        # Zang
        test_vs = 10.915* self.cpt.tip**0.317 * self.cpt.IC**0.21*  self.cpt.depth**0.057*0.92
        test_GO = self.cpt.rho * test_vs ** 2
        self.cpt.vs_calc(method="Zang")
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)
        # Ahmed
        test_vs = 1000*np.e**(-0.887* self.cpt.IC)*(1+0.443*self.cpt.Fr*self.cpt.effective_stress/100*9.81/self.cpt.gamma)**0.5
        test_GO = self.cpt.rho * test_vs ** 2
        self.cpt.vs_calc(method="Ahmed")
        self.assertEqual(test_vs[0], self.cpt.vs[0])
        np.testing.assert_array_equal(test_GO, self.cpt.G0)
        # All
        self.cpt.vs_calc(method="all")
        import os
        self.assertTrue(os.path.isfile("UNIT_TESTING_shear_modulus.png"))
        self.assertTrue(os.path.isfile("UNIT_TESTING_shear_wave.png"))
        return

    def test_poisson_calc(self):
        self.cpt.lithology = ['1','2','3','4','5']
        test_poisson = [0.5,0.5,0.5,0.2,0.2]
        self.cpt.poisson_calc()
        np.testing.assert_array_equal(test_poisson, self.cpt.poisson)
        return
    def test_damp_calc(self):
        self.cpt.lithology = ['1','2','3','4','5','6','7','8','9',]
        test_damping = np.zeros(9)
        self.cpt.damp_calc()
        np.testing.assert_array_equal(test_damping, self.cpt.damping)
        return
    def test_qt_calc(self):
        self.cpt.tip =np.array( [1])
        self.cpt.water = np.array( [1])
        self.cpt.a =np.array( [1])
        test_qt = np.array( [1])
        self.cpt.qt_calc()
        np.testing.assert_array_equal(test_qt, self.cpt.qt)
        return

    def test_write_csv(self):
        self.cpt.name = 'UNIT_TEST'
        self.cpt.write_csv()
        import os.path
        self.assertTrue(os.path.isfile('UNIT_TEST.csv'))
        return
    def test_plot_lithology(self):
        self.cpt.name = 'UNIT_TEST'
        self.cpt.plot_lithology()
        import os.path
        self.assertTrue(os.path.isfile('UNIT_TEST_lithology.png'))
        return

    def test_plot_cpt(self):
        self.cpt.name = 'UNIT_TEST'
        self.cpt.plot_cpt()
        import os.path
        self.assertTrue(os.path.isfile('UNIT_TEST_cpt.png'))
        return

    def test_plot_correlations(self):
        self.cpt.name = 'UNIT_TEST'
        self.cpt.plot_correlations([],'Correlations' , 'Correlations', 'Correlations')
        import os.path
        self.assertTrue(os.path.isfile('UNIT_TEST_Correlations.png'))
        return

    def test_add_json(self):
        jsn = {"scenarios": []}
        i = 0
        self.cpt.coord = [1, 2]
        self.cpt.add_json(jsn, i)
        # check if coordinates have been added
        self.assertEqual(jsn['scenarios'][0]['coordinates'][0], self.cpt.coord[0])
        self.assertEqual(jsn['scenarios'][0]['coordinates'][1], self.cpt.coord[1])
        return

    def test_dump_json(self):
        import os
        jsn = {"scenarios": []}
        jsn["scenarios"].append({"coordinates": [1, 2]})

        input_dic = {"Source_x": 1,
                     "Source_y": 1,
                     "Receiver_x": 1,
                     "Receiver_y": 1,
                     }
        self.cpt.update_dump_json(jsn, input_dic)
        # check if probability is 100
        self.assertEqual(jsn['scenarios'][0]['probability'], 100)
        # check if file has been created
        self.assertTrue(os.path.isfile("./results.json"))
        os.remove("./results.json")
        return

    def test_compute_prob(self):
        cpt_coord = np.array([[0, 10], [17, 19], [14, 22], [35, 10]])
        source_coord = [5, 10]
        receiver_coord = [30, 10]
        prob = cpt_module.compute_probability(cpt_coord, source_coord, receiver_coord)
        # exact results
        exact = [0.306964061716986,
                 0.263692716163477,
                 0.254225518484290,
                 0.175117703635247]

        self.assertAlmostEqual(prob[0], exact[0]*100)
        self.assertAlmostEqual(prob[1], exact[1]*100)
        self.assertAlmostEqual(prob[2], exact[2]*100)
        self.assertAlmostEqual(prob[3], exact[3]*100)

        return

    def tearDown(self):
        import os
        self.log_file.close()
        os.remove("./results/log_file.txt")
        return


if __name__ == '__main__':  # pragma: no cover
    from teamcity import is_running_under_teamcity
    if is_running_under_teamcity():
        from teamcity.unittestpy import TeamcityTestRunner
        runner = TeamcityTestRunner()
    else:
        runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
