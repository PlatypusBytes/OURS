class CPT:
    r"""
    CPT module

    Read and process cpt files: GEF format

    """

    def __init__(self, out_fold, log_file):
        import os
        # variables
        self.depth = []
        self.coord = []
        self.NAP = []
        self.tip = []
        self.friction = []
        self.friction_nbr = []
        self.name = []
        self.gamma = []
        self.rho = []
        self.total_stress = []
        self.effective_stress = []
        self.qt = []
        self.Qtn = []
        self.Fr = []
        self.IC = []
        self.n = []
        self.vs = []
        self.G0 = []
        self.poisson = []
        self.damping = []
        self.water = []
        self.lithology = []
        self.litho_points = []
        # self.lithology_simple = []
        self.lithology_json = []
        self.depth_json = []
        self.indx_json = []
        self.unit_testing = False

        # checks if file_path exits. If not creates file_path
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)
        self.output_folder = out_fold

        # fixed values
        self.g = 9.81
        self.Pa = 100.
        self.a = 0.8

        # log file
        self.log_file = log_file

        return

    def read_gef(self, gef_file, key_cpt):
        r"""
        Reads CPT gef file. Returns a structure.

        Parameters
        ----------
        :param gef_file: file path and file name to the CPT gef file
        :param key_cpt: dictionary with the CPT key
        """

        import os
        import numpy as np
        import re

        with open(gef_file, 'r', encoding="Ansi") as f:
            data = f.readlines()

        gef_file_name = os.path.split(gef_file)[-1].upper().split(".GEF")[0]

        # search NAP
        try:
            idx_nap = [i for i, val in enumerate(data) if val.startswith(r'#ZID=')][0]
            NAP = float(data[idx_nap].split(',')[1])
        except IndexError:
            self.log_file.error_message("file " + gef_file_name + " contains no NAP position")
            return False
        # search end of header
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r'#EOH=')][0]
        # search for coordinates
        try:
            idx_coord = [i for i, val in enumerate(data) if val.startswith(r'#XYID=')][0]
        except IndexError:
            self.log_file.error_message("file " + gef_file_name + " contains no coordinates")
            return False
        # check if coordinates are non-zero
        if all(i == 0 for i in list(map(float, data[idx_coord].split(',')[1:3]))):
            self.log_file.error_message("file " + gef_file_name + " has coordinates 0, 0")
            return False
        # search index depth
        try:
            idx_depth = [int(val.split(',')[0].split("=")[-1]) - 1 for i, val in enumerate(data)
                         if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['depth'])][0]
        except IndexError:
            self.log_file.error_message("file " + gef_file_name + " contains no length")
            return False
        # search index tip resistance
        try:
            idx_tip = [int(val.split(',')[0].split("=")[-1]) - 1 for i, val in enumerate(data)
                       if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['tip'])][0]
        except IndexError:
            self.log_file.error_message("file " + gef_file_name + " contains no tip")
            return False
        # search index friction
        try:
            idx_friction = [int(val.split(',')[0].split("=")[-1]) - 1 for i, val in enumerate(data)
                            if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['friction'])][0]
        except IndexError:
            self.log_file.error_message("file " + gef_file_name + " contains no friction")
            return False
        # search index friction number
        try:
            idx_friction_nb = [int(val.split(',')[0].split("=")[-1]) - 1 for i, val in enumerate(data)
                               if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['friction_nb'])][0]
        except IndexError:
            self.log_file.error_message("file " + gef_file_name + " contains no friction number")
            return False
        # search index water if water sensor is available:
        try:
            idx_water = [int(val.split(',')[0].split("=")[-1]) - 1 for i, val in enumerate(data) if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['water'])][0]
        except IndexError:
            idx_water = False
        # rewrite data with separator ;
        data[idx_EOH + 1:] = [re.sub("[ ,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1:]]

        # if first column is -9999 it is error and is is neglected
        if float(data[idx_EOH + 1].split(";")[1]) == -9999:
            idx_EOH += 1

        # read data & correct depth to NAP
        depth_tmp = [float(data[i].split(";")[idx_depth]) for i in range(idx_EOH + 1, len(data))]
        tip_tmp = [float(data[i].split(";")[idx_tip]) * 1000. for i in range(idx_EOH + 1, len(data))]
        friction_tmp = [float(data[i].split(";")[idx_friction]) * 1000. for i in range(idx_EOH + 1, len(data))]
        friction_nb_tmp = [float(data[i].split(";")[idx_friction_nb]) for i in range(idx_EOH + 1, len(data))]
        if idx_water:
            water_tmp = [float(data[i].split(";")[idx_water]) * 1000 for i in range(idx_EOH + 1, len(data))]
        else:
            water_tmp = np.zeros(len(depth_tmp))

        # if tip / friction / friction number are negative -> zero
        tip_tmp = np.array(tip_tmp)
        tip_tmp[tip_tmp < 0] = 0
        friction_tmp = np.array(friction_tmp)
        friction_tmp[friction_tmp < 0] = 0
        friction_nb_tmp = np.array(friction_nb_tmp)
        friction_nb_tmp[friction_nb_tmp < 0] = 0
        # if "water" in key_cpt.keys():
        water_tmp = np.array(water_tmp)
        water_tmp[water_tmp < 0] = 0

        # remove the points with error: value == -9999
        idx = [[i for i, d in enumerate(depth_tmp) if d == -9999],
               [i for i, d in enumerate(tip_tmp) if d == -9999],
               [i for i, d in enumerate(friction_tmp) if d == -9999],
               [i for i, d in enumerate(friction_nb_tmp) if d == -9999],
               [i for i, d in enumerate(water_tmp) if d == -9999]]

        idx_remove = [item for sublist in idx for item in sublist]

        # assign variables
        self.name = gef_file_name
        self.coord = list(map(float, data[idx_coord].split(',')[1:3]))

        self.depth = np.array([np.abs(i) for j, i in enumerate(depth_tmp) if j not in idx_remove])
        self.depth -= self.depth[0]
        self.NAP = np.array([-i + NAP for j, i in enumerate(depth_tmp) if j not in idx_remove])
        self.tip = np.array([i for j, i in enumerate(tip_tmp) if j not in idx_remove])
        self.friction = np.array([i for j, i in enumerate(friction_tmp) if j not in idx_remove])
        self.friction_nbr = np.array([i for j, i in enumerate(friction_nb_tmp) if j not in idx_remove])
        self.water = np.array([i for j, i in enumerate(water_tmp) if j not in idx_remove])

        return True

    def lithology_calc(self):
        r"""
        Lithology calculation.

        Computes the lithology following Robertson and Cabal [1]_.

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014.
        """
        import robertson

        classification = robertson.Robertson()
        classification.soil_types()

        # compute Qtn and Fr
        self.norm_calc()

        litho, points = classification.lithology(self.Qtn, self.Fr)

        # assign to variables
        self.lithology = litho
        self.litho_points = points

        # # compute simplified lithology
        # # group the following zones
        # # Zones 1 & 2 & 3 -> A
        # # Zones 4 & 5 & 6 -> B
        # # Zones 7 & 8 & 9 -> C
        # litho = ["A" if x=="1" or x == "2" or x == "3" else x for x in litho]
        # litho = ["B" if x=="4" or x == "5" or x == "6" else x for x in litho]
        # litho = ["C" if x=="7" or x == "8" or x == "9" else x for x in litho]
        #
        # self.lithology_simple = litho

        return

    def lithology_calc_iter(self, gamma_limit, z_pwp, iter_max=100):
        r"""
        Lithology calculation.

        Computes the lithology following Robertson and Cabal [1]_.

        Parameters
        ----------
        :param gamma_limit: Maximum value for gamma
        :param z_pwp: Depth pore water pressure
        :param iter_max: (optional) Maximum number of iterations

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014.
        """
        import robertson
        import numpy as np 		

        classification = robertson.Robertson()
        classification.soil_types()

        # assumption: -> all is sand
        litho = ["9"] * len(self.NAP)
        # qt = robertson.qt_calc(litho, self.tip, self.water, self.a)
        self.qt_calc(litho)
        qt = self.qt

        # compute gamma
        self.gamma_calc(gamma_limit)
        # compute stress
        self.stress_calc(z_pwp)
        # compute Qtn and Fr
        self.norm_calc()

        litho, points = classification.lithology(self.Qtn, self.Fr)

        iter = 0
        # if pore water pressures exist -> do interactive process
        if np.size(self.water) > 0:
            # iterative process to update qt
            self.qt_calc(litho)
            qt2 = self.qt
            while not np.array_equal(qt, qt2):
                # compute gamma
                self.gamma_calc(gamma_limit)
                # compute stress
                self.stress_calc(z_pwp)
                # compute Qtn and Fr
                self.norm_calc()

                litho, points = classification.lithology(self.Qtn, self.Fr)
                qt = qt2
                self.qt_calc(litho)
                qt2 = self.qt
                iter += 1
                if iter == iter_max:
                    print("Classification reach maximum number of iterations: " + str(iter_max))
                    break
        # assign to variables
        self.lithology = litho
        self.litho_points = points

        # # compute simplified lithology
        # # group the following zones
        # # Zones 1 & 2 & 3 -> A
        # # Zones 4 & 5 & 6 -> B
        # # Zones 7 & 8 & 9 -> C
        # litho = ["A" if x=="1" or x == "2" or x == "3" else x for x in litho]
        # litho = ["B" if x=="4" or x == "5" or x == "6" else x for x in litho]
        # litho = ["C" if x=="7" or x == "8" or x == "9" else x for x in litho]
        #
        # self.lithology_simple = litho

        return

    def gamma_calc(self, gamma_limit, method="Robertson"):
        r"""
        Computes unit weight.

        Computes the unit weight following Robertson and Cabal [1]_.
        If unit weight is infinity, it is set to gamma_limit.
        The formula for unit weight is:

        .. math::

            \gamma = 0.27 \log(R_{f}) + 0.36 \log\left(\frac{q_{t}}{Pa}\right) + 1.236

        Alternative method of Lengkeek et al. [2]

        .. math::

            \gamma = \gamma_{sat,ref} - \beta \left( \frac{\log \left( \frac{q_{t,ref}}{q_{t}} \right)}{\log \left(\frac{R_{f,ref}}{R_{f}}\right)} \right)

        Parameters
        ----------
        :param gamma_limit: Maximum value for gamma
        :param method: (optional) Method to compute unit weight. Default is Robertson

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 36.
        .. [2] Lengkeek, A., de Greef, J., & Joosten, S. *CPT based unit weight estimation extended to soft organic soils and peat.* Proceedings of the 4th International Symposium on Cone Penetration Testing (CPT'18), 2018, pp: 389-394.
        """
        import numpy as np
        np.seterr(divide="ignore")

        # calculate unit weight according to Robertson & Cabal 2015
        if method == "Robertson":
            aux = 0.27 * np.log10(self.friction_nbr) + 0.36 * np.log10(self.qt / self.Pa) + 1.236
            aux[np.abs(aux) == np.inf] = gamma_limit / self.g
            aux[aux < 0] = 0.
            self.gamma = aux * self.g
        elif method == "Lengkeek":
            aux = 19. - 4.12 * np.log10(5000. / self.qt) / np.log10(30. / self.friction_nbr)
            aux[np.abs(aux) == np.inf] = gamma_limit
            aux[aux < 0] = 0.
            self.gamma = aux
        elif method == "all":  # if all, compares all the methods and plot
            self.gamma_calc(gamma_limit, method="Lengkeek")
            gamma_1 = self.gamma
            self.gamma_calc(gamma_limit, method="Robertson")
            gamma_2 = self.gamma
            self.plot_correlations([gamma_1, gamma_2], "Unit Weight [kN/m3]", ["Lengkeek", "Robertson"], "unit_weight")
            pass
        return

    def rho_calc(self):
        r"""
        Computes density of soil.

        The formula for density is:

        .. math::

            \rho = \frac{\gamma}{g}
        """

        self.rho = self.gamma * 1000. / self.g
        return

    def stress_calc(self, z_pwp):
        r"""
        Computes total and effective stress

        Parameters
        ----------
        :param z_pwp: Depth of pore water pressure in NAP
        """
        # compute total and effective stress
        import numpy as np 		

        # compute depth diff
        z = np.diff(np.abs((self.depth - self.depth[0])))
        z = np.append(z, z[-1])
        # total stress
        self.total_stress = np.cumsum(self.gamma * z) + self.depth[0] * np.mean(self.gamma[:10])
        # compute pwp
        # determine location of phreatic line: it cannot be above the CPT depth
        z_aux = np.min([z_pwp, self.NAP[0] + self.depth[0]])
        pwp = (z_aux - self.NAP) * self.g
        # no suction is allowed
        pwp[pwp <= 0] = 0
        # compute effective stress
        self.effective_stress = self.total_stress - pwp
        # if effective stress is negative -> effective stress = 0
        self.effective_stress[self.effective_stress <= 0] = 0

        return

    def norm_calc(self, n_method=False):
        r"""
        normalisation of qc and friction into Qtn and Fr, following Robertson and Cabal [1]_.

        .. math::

            Q_{tn} = \left(\frac{q_{t} - \sigma_{v0}}{Pa} \right) \left(\frac{Pa}{\sigma_{v0}'}\right)^{n}

            F_{r} = \frac{f_{s}}{q_{t}-\sigma_{v0}} \cdot 100


        Parameters
        ----------
        :param n_method: (optional) parameter *n* stress exponent. Default is n computed in an iterative way.

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 108.
        """

        # normalisation of qc and friction into Qtn and Fr: following Robertson and Cabal (2015)
        import numpy as np 		

        # iteration around n to compute IC
        # start assuming n=1 for IC calculation
        n = np.ones(len(self.tip))

        # switch for the n calculation. default is iterative process
        if not n_method:
            tol = 1.e-12
            error = 1
            max_ite = 10000
            itr = 0
            while error >= tol:
                # if did not converge
                if itr >= max_ite:
                    n = np.ones(len(self.tip)) * 0.5
                    break
                n1 = n_iter(n, self.tip, self.friction_nbr, self.effective_stress, self.total_stress, self.Pa)
                error = np.linalg.norm(n1 - n) / np.linalg.norm(n1)
                n = n1
                itr += 1
        else:
            n = np.ones(len(self.tip)) * 0.5

        # parameter Cn
        Cn = (self.Pa / self.effective_stress) ** n
        # calculation Q and F
        Q = (self.tip - self.total_stress) / self.Pa * Cn
        F = self.friction / (self.tip - self.total_stress) * 100
        # Q and F cannot be negative. if negative, log10 will be infinite.
        # These values are limited by the contours of soil behaviour of Robertson
        Q[Q <= 1.] = 1.
        F[F <= 0.1] = 0.1
        Q[Q >= 1000.] = 1000.
        F[F >= 10.] = 10.
        self.Qtn = Q
        self.Fr = F
        self.n = n

        return

    # def qc1n_calc(self):
    #     r"""
    #     Normalisation of qc into qc1n
    #
    #     Normalisation of qc into qc1n following Boulanger and Idriss [2]_.
    #
    #     .. math::
    #
    #         q_{c1N} = C_{N} \cdot \frac{q_{c}}{Pa}
    #
    #     .. rubric:: References
    #     .. [2] Boulanger, R.W. and Idriss, I.M. *CPT and SPT based liquefaction triggering procedures.* UC Davis, 2014, pg: 6.
    #     """
    #     import numpy as np 		
    #
    #     # normalise qc
    #     qc100 = self.tip * np.min([np.ones(len(self.tip)) * 1.7, (self.Pa / self.effective_stress)**self.n], axis=0)
    #     self.qc1n = qc100 / self.Pa
    #     return

    def IC_calc(self):
        r"""
        IC, following Robertson and Cabal [1]_.

        .. math::

            I_{c} = \left[ \left(3.47 - \log\left(Q_{tn}\right) \right)^{2} + \left(\log\left(F_{r}\right) + 1.22 \right)^{2} \right]^{0.5}

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 104.
        """

        # IC: following Robertson and Cabal (2015)
        import numpy as np 		
        # compute IC
        self.IC = ((3.47 - np.log10(self.Qtn)) ** 2. + (np.log10(self.Fr) + 1.22) ** 2.) ** 0.5
        return

    def vs_calc(self, method="Robertson"):
        r"""
        Shear wave velocity and shear modulus. The following methods are available:

        * Robertson and Cabal [1]_:

        .. math::

            v_{s} = \left( \alpha_{vs} \cdot \frac{q_{t} - \sigma_{v0}}{Pa} \right)^{0.5}

            \alpha_{vs} = 10^{0.55 I_{c} + 1.68}

            G_{0} = \frac{\gamma}{g} \cdot v_{s}^{2}

        * Mayne [3]_:

        .. math::

            v_{s} = e^{\frac{\gamma_{sat} + 4.03}{4.17}} \cdot \left( \frac{\sigma_{v0}'}{\sigma_{atm}} \right)^{0.25}

            v_{s} = 118.8 \cdot \log \left(f_{s} \right) + 18.5

        * Andrus et al. [4]_:

        .. math::

            v_{s} = 2.27 \cdot q_{t}^{0.412} \cdot I_{c}^{0.989} \cdot D^{0.033} \cdot ASF  (Holocene)

            v_{s} = 2.62 \cdot q_{t}^{0.395} \cdot I_{c}^{0.912} \cdot D^{0.124} \cdot SF   (Pleistocene)

        * Zang & Tong [5]_:

        .. math::

            v_{s} = 10.915 \cdot q_{t}^{0.317} \cdot I_{c}^{0.210} \cdot D^{0.057} \cdot SF^{a}  (Holocene)

        * Ahmed [6]_:

        .. math::

            v_{s} = 1000 \cdot e^{-0.887 \cdot I_{c}} \cdot \left( \left(1 + 0.443 \cdot F_{r} \right) \cdot \left(\frac{\sigma'_{v}}{p_{a}} \right) \cdot \left(\frac{\gamma_{w}}{\gamma} \right) \right)^{0.5}


        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 48-50.
        .. [3] Mayne, P.W. *In-Situ Test Calibrations for Evaluating Soil Parameters.* Characterisation and enginering properties of natural soils, Volume 3.
               2006, pg: 1-56.
        .. [4] Andrus, R.D., Mohanan, N.P., Piratheepan, P., et al. *Predicting shear-wave velocity from cone penetration resistance.* Proceedings, 4th International Conference on Earthquake Geotechnical Engineering. 2007.
        .. [5] Zang, M. & Tong, L. *New statistical and graphical assessment of CPT-based empirical correlations for the shear wave velocity of soils* Engineering Geology 226 (2017) 184–191
        .. [6] Ahmed, S.M. *Correlating the Shear Wave Velocity with the Cone Penetration Test.* Proceedings of the 2nd World Congress on Civil, Structural, and Environmental Engineering, 2017 ,page 4.
        """
        import numpy as np

        if method == "Robertson":
            # vs: following Robertson and Cabal (2015)
            alpha_vs = 10 ** (0.55 * self.IC + 1.68)
            aux = alpha_vs * (self.qt - self.total_stress) / self.Pa
            aux[aux < 0] = 0
            self.vs = aux**0.5
            self.G0 = self.rho * self.vs**2
        elif method == "Mayne":
            # vs: following Mayne (2006)
            self.vs = np.exp((self.gamma + 4.03) / 4.17) * (self.effective_stress / self.Pa) ** 0.25
            self.G0 = self.rho * self.vs ** 2
        elif method == "Andrus":
            # vs: following Andrus (2007)
            self.vs = 2.27 * self.qt ** 0.412 * self.IC ** 0.989 * self.depth ** 0.033 * 1
            self.G0 = self.rho * self.vs ** 2
        elif method == "Zang":
            # vs: following Zang & Tong (2017)
            self.vs = 10.915 * self.qt ** 0.317 * self.IC ** 0.210 * self.depth ** 0.057 * 0.92
            self.G0 = self.rho * self.vs ** 2
        elif method == "Ahmed":
            self.vs = 1000. * np.exp(-0.887 * self.IC) * (1. + 0.443 *self.Fr * self.effective_stress / self.Pa * self.g / self.gamma) ** 0.5
            self.G0 = self.rho * self.vs ** 2
        elif method == "all":  # compares all and assumes default
            self.vs_calc(method="Mayne")
            vs1 = self.vs
            G0_1 = self.G0
            self.vs_calc(method="Andrus")
            vs3 = self.vs
            G0_3 = self.G0
            self.vs_calc(method="Zang")
            vs4 = self.vs
            G0_4 = self.G0
            self.vs_calc(method="Ahmed")
            vs5 = self.vs
            G0_5 = self.G0
            self.vs_calc(method="Robertson")
            vs2 = self.vs
            G0_2 = self.G0
            self.plot_correlations([vs1, vs2, vs3, vs4, vs5], "Shear wave velocity [m/s]", ["Mayne", "Robertson", "Andrus", "Zang", "Ahmed"], "shear_wave")
            self.plot_correlations([G0_1, G0_2, G0_3, G0_4, G0_5], "Shear modulus [kPa]", ["Mayne", "Robertson", "Andrus", "Zang", "Ahmed"], "shear_modulus")
            pass

        return

    def damp_calc(self, d_min=2, Cu=2., D50=0.2, Ip=40., method="Mayne"):
        r"""
        Damping calculation.

        For clays and peats, the damping is assumed as the minimum damping following Darendeli [7]_.

        .. math::

            D_{min} = \left(0.8005 + 0.0129 \cdot PI \cdot OCR^{-0.1069} \right) \cdot \sigma_{v0}'^{-0.2889} \cdot \left[ 1 + 0.2919 \ln \left( freq \right) \right]

        The OCR can be computed according to Mayne [1]_ or Robertson [2]_.


        .. math::

            OCR_{Mayne} = 0.33 \cdot \frac{q_{t} - \sigma_{v0}}{\sigma_{v0}'}

            OCR_{Rob} = 0.25 \left(Q_{t}\right)^{1.25}


        For sand the damping is assumed as the minimum damping following Menq [8]_.

        .. math::
            D_{min} = 0.55 \cdot C_{u}^{0.1} \cdot d_{50}^{-0.3} \cdot  \left(\frac{\sigma'_{v}}{p_{a}} \right)^-0.08


        Parameters
        ----------
        :param d_min: (optional) Minimum damping. Default is 2%
        :param Cu: (optional) Coefficient of uniformity. Default is 2.0
        :param D50: (optional) Median grain size. Default is 0.2 mm
        :param Ip: (optional) Plasticity index. Default is 40
        :param method: (optional) Method for calculation of OCR. Default is Mayne


        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 40.
        .. [2] Mayne, P. *Cone Penetration Testing. A Synthesis of Highway Practice.* Transportation Research Board, 2007, pg: 34.
        .. [7] Darendeli, M.B. *Development of a New Family of Normalized Modulus Reduction and material damping curves.* PhD thesis, 2001, pg: 221.
        .. [8] Menq, F.Y. *Dynamic Properties of Sandy and Gravelly Soils.* PhD Thesis, 2003, Department of Civil Engineering, University of Texas, Austin, TX.
        """
        # ToDo missing frequency dependency
        import numpy as np

        # assign size to damping
        self.damping = np.zeros(len(self.lithology)) + d_min
        OCR = np.zeros(len(self.lithology))

        for i, lit in enumerate(self.lithology):
            # if  clay
            if lit == "3" or lit == "4":
                if method == "Mayne":
                    OCR[i] = 0.33 * (self.qt[i] - self.total_stress[i]) / self.effective_stress[i]
                elif method == "Robertson":
                    OCR[i] = 0.25 * self.Qtn[i] ** 1.25
                    # OCR[i] = 0.25 * ((self.qt[i] - self.total_stress[i]) / self.effective_stress[i]) ** 1.25
                self.damping[i] = (0.8005 + 0.129 * Ip * OCR[i] ** -0.1069) * (self.effective_stress[i] / self.Pa) ** -0.2889
            # if sand:
            elif lit == "5" or lit == "6" or lit == "7":
                self.damping[i] = 0.55 * Cu ** 0.1 * D50 ** -0.3 * (self.effective_stress[i] / self.Pa) ** -0.08
            # if peat:
            elif lit == "2":
                # same as clay: OCR=1 IP=100
                self.damping[i] = 2.512 * (self.effective_stress[i] / self.Pa) ** -0.2889
        return

    def poisson_calc(self):
        r"""
        Poisson ratio. Following [2]_.

        Poisson assumed 0.5 for soft layers and 0.2 for sandy layers.

        .. rubric:: References
        .. [2] Mayne, P. *Cone Penetration Testing. A Synthesis of Highway Practice.* Transportation Research Board, 2007, pg: 31.
        """
        import numpy as np 		

        # assign size to poisson
        self.poisson = np.zeros(len(self.lithology))

        for i, lit in enumerate(self.lithology):
            # if soft layer
            if lit == "1" or lit == "2" or lit == "3":
                self.poisson[i] = 0.5
            else:
                self.poisson[i] = 0.2
        return

    def qt_calc(self):
        r"""
        Corrected cone resistance, following Robertson and Cabal [1]_.

        .. math::

            q_{t} = q_{c} + u_{2} \left( 1 - a\right)

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 22.
        """

        # qt computed following Robertson & Cabal (2015)
        # qt = qc + u2 * (1 - a)

        self.qt = self.tip + self.water * (1. - self.a)
        return

    def merge_thickness(self, min_layer_thick):
        """
        Reorganises the lithology based on the minimum layer thickness.

        Parameters
        ----------
        :param min_layer_thick: Minimum layer thickness


        """
        import numpy as np


        depth = self.depth
        lithology = self.lithology
        # find location of the start of the soil types
        aux = ""
        idx = []
        local_IC = []
        # here the i is where I am in the Cpt , val is the value of the lithology which is a number coresponding to Robertson
        # if the previous layer is not the same as the next save the position and save the value for the next step
        for j, val in enumerate(lithology):
            if val != aux:
                aux = val
                idx.append(j)
        target_idx = idx[1:]
        for i in range(len(idx)):
            if i == len(idx)-1:
                local_IC.append(np.mean(self.IC[idx[i]:]))
            else:
                local_IC.append(np.mean(self.IC[idx[i]:target_idx[i]]))

        local_z_ini = [depth[i] for i in idx]
        # thickness between those layers calculated
        local_thick = list(np.diff(local_z_ini)),[depth[-1] - local_z_ini[-1]]
        local_thick = sum(local_thick,[])
        # soil type
        local_label = [lithology[i] for i in idx]

        IC_checker = [['next','back'][abs(local_IC[i]-local_IC[i+1]) > abs(local_IC[i-1]-local_IC[i])] for i in range(1,len(local_IC)-1)]
        IC_checker.insert(0,'next')
        IC_checker.append('back')
        now_thickness = 0
        new_thickness = np.zeros(len(idx))
        for counter,value in enumerate(local_thick):
            if counter is not 0 and now_thickness < float(min_layer_thick):
                now_thickness = now_thickness + value
            else:
                now_thickness = value
            new_thickness[counter]= now_thickness
        min_checker = [[False, True][x >= float(min_layer_thick)] for x in new_thickness]
        # has to be changed to append
        for counter in range(1,len(new_thickness)):
            if IC_checker[counter] == 'back' and min_checker[counter-1] == True:
                new_thickness[counter] = new_thickness[counter-1] + new_thickness[counter]
            else:
                pass

        new_index , new_depth , new_label = [] , [] , []
        last_time = -1
        for counter,value in enumerate(idx[1:]):
            if min_checker[counter]== True :
                new_index.append(value)
                new_depth.append(local_z_ini[counter+1])
                new_label.append('/'.join(local_label[last_time+1:counter+1]))
                last_time = counter
        new_index.insert(0,idx[0])
        new_depth.insert(0,self.depth[0])
        if new_index[-1] is not idx[-1]:
            new_label.append('/'.join(local_label[last_time+1:]))
        self.lithology_json = new_label
        self.depth_json = new_depth
        self.indx_json = new_index

        return

    def add_json(self, jsn, id):
        """
        Add to json file the results.

        Parameters
        ----------
        :param jsn: Json data structure
        :param id: Scenario (index)
        """
        import numpy as np 		

        # create data
        data = {"lithology": [],
                "depth": [],
                "E": [],
                "v": [],
                "rho": [],
                "damping": [],
                "var_E": [],
                "var_v": [],
                "var_rho": [],
                "var_damping": [],
                }

        # populate structure
        for i in range(len(self.indx_json) - 1):
            data["lithology"].append(str(self.lithology_json[i]))
            data["depth"].append(self.depth_json[i])
            E = 2 * (1 + 0.3) * self.gamma[self.indx_json[i]:self.indx_json[i + 1]] * 100 * \
                self.vs[self.indx_json[i]: self.indx_json[i + 1]]**2

            data["E"].append(np.mean(E))
            data["var_E"].append(np.std(E))

            poisson = self.poisson[self.indx_json[i]:self.indx_json[i + 1]]

            data["v"].append(np.mean(poisson))
            data["var_v"].append(np.std(poisson))

            rho = self.rho[self.indx_json[i]:self.indx_json[i + 1]]
            data["rho"].append(np.mean(rho))
            data["var_rho"].append(np.std(rho))

            # ToDo update damping
            damp = self.damping[self.indx_json[i]:self.indx_json[i + 1]]
            data["damping"].append(np.mean(damp))
            data["var_damping"].append(np.std(damp))

        jsn["scenarios"].append({"Name": "Scenario " + str(id + 1)})
        jsn["scenarios"][id].update({"coordinates": self.coord,
                                     "probability": [],
                                     "data": data})

        return

    def update_dump_json(self, jsn, input_dic):
        """

        Computes the probability of the scenario and dump json file into output file.

        Parameters
        ----------
        :param jsn: json file with data structure
        :param input_dic: dictionary with the input information
        :return:
        """
        import os
        import json

        # update the probability
        coord_source = [float(input_dic["Source_x"]), float(input_dic["Source_y"])]
        coord_receiver = [float(input_dic["Receiver_x"]), float(input_dic["Receiver_y"])]
        coord_cpts = [i['coordinates'] for i in jsn["scenarios"]]
        probs = compute_probability(coord_cpts, coord_source, coord_receiver)

        # update the json file
        for i in range(len(jsn["scenarios"])):
            jsn["scenarios"][i]["probability"] = probs[i]

        # write file
        with open(os.path.join(self.output_folder, "results.json"), "w") as fo:
            json.dump(jsn, fo, indent=4)
        return

    def plot_correlations(self, x_data, x_label, l_name, name):
        """
        Plot CPT correlations.

        Parameters
        ----------
        :param x_data: dataset for the plots
        :param x_label: label of the plots
        :param l_name: name of the different correlations within the dataset
        :param name: name for the output file
        """
        import os
        import numpy as np
        import matplotlib.pylab as plt
        from cycler import cycler

        # data
        y_data = self.depth
        y_label = "Depth [m]"

        # set the color list
        colormap = plt.cm.gist_ncar
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(x_data))]
        plt.gca().set_prop_cycle(cycler('color', colors))
        plt.figure(figsize=(4, 6))

        # plot for each y_value
        for i in range(len(x_data)):
            plt.plot(x_data[i], y_data, label=l_name[i])

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid()
        plt.legend(loc=1, prop={'size': 12})
        # invert y axis
        plt.gca().invert_yaxis()
        plt.tight_layout()
        # save the figure
        plt.savefig(os.path.join(self.output_folder, self.name + "_" + name) + ".png")
        plt.close()
        return

    def plot_cpt(self, nb_plots=6):
        """
        Plot CPT values.

        Parameters
        ----------
        :param nb_plots: (optional) number of plots
        """
        import os
        import numpy as np 		
        import matplotlib.pylab as plt
        from cycler import cycler

        # data
        x_data = [self.tip, self.friction_nbr, self.rho, self.G0, self.poisson, self.damping]
        y_data = self.depth
        l_name = ["Tip resistance", "Friction number", "Density", "Shear modulus", "Poisson ratio", "Damping"]
        x_label = ["Tip resistance [kPa]", "Friction number [-]", r"Density [kg/m$^{3}$]", "Shear modulus [kPa]", "Poisson ratio [-]", "Damping [%]"]
        y_label = "Depth [m]"

        # set the color list
        colormap = plt.cm.gist_ncar
        colors = [colormap(i) for i in np.linspace(0, 0.9, nb_plots)]
        plt.gca().set_prop_cycle(cycler('color', colors))
        plt.subplots(1, nb_plots, figsize=(20, 6))

        # plot for each y_value
        for i in range(nb_plots):
            plt.subplot(1, nb_plots, i + 1)
            plt.plot(x_data[i], y_data, label=l_name[i])

            # plt.title(title)
            plt.xlabel(x_label[i], fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid()
            plt.legend(loc=1, prop={'size': 12})
            # invert y axis
            plt.gca().invert_yaxis()

        plt.tight_layout()
        # save the figure
        plt.savefig(os.path.join(self.output_folder, self.name) + "_cpt.png")
        plt.close()

        return

    def plot_lithology(self):
        """
        Plot CPT lithology.

        """
        import os
        import numpy as np 		
        import matplotlib.pylab as plt
        import matplotlib.patches as patches

        # define figure
        f, (ax1, ax3) = plt.subplots(1, 2, figsize=(7, 10), sharey=True)

        # first subplot - CPT
        # ax1 tip
        ax1.set_position([0.15, 0.1, 0.4, 0.8])
        ax1.plot(self.tip, self.depth, label="Tip resistance", color="b")
        ax1.set_xlabel("Tip resistance [kPa]", fontsize=12)
        ax1.set_ylabel("Depth [m]", fontsize=12)
        ax1.set_xlim(left=0)
        # invert y axis
        ax1.invert_yaxis()

        # ax2 friction number
        ax2 = ax1.twiny()
        ax2.set_position([0.15, 0.1, 0.4, 0.8])
        ax2.plot(self.friction_nbr, self.depth, label="Friction number", color="r")
        ax2.set_xlabel("Friction number [-]", fontsize=12)
        ax2.set_xlim(left=0)

        # align grid
        ax1.set_xticks(np.linspace(0, ax1.get_xticks()[-1], 5))
        ax2.set_xticks(np.linspace(0, ax2.get_xticks()[-1], 5))

        ax1.grid()

        # second subplot - Lithology
        litho = np.array(self.lithology).astype(int)
        color_litho = ["red", "brown", "cyan", "blue", "gray", "yellow", "orange", "green", "lightgray"]

        ax3.set_position([0.60, 0.1, 0.05, 0.8])
        ax3.set_xlim((0, 10.))
        # remove ticks
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.tick_params(axis='both', which='both', length=0)
        # ax3.set_xlabel("Lithology", fontsize=12)
        # ax3.xaxis.set_label_position('top')

        # thickness
        diff_depth = np.diff(self.depth)

        # color the field
        for i in range(len(self.depth) - 1):
            ax3.add_patch(patches.Rectangle(
                (0, self.depth[i]),
                10.,
                diff_depth[i],
                fill=True,
                color=color_litho[litho[i] - 1]))

        # create legend for Robertson
        for i, c in enumerate(color_litho):
            ax3.add_patch(patches.Rectangle(
                (16, ax1.get_ylim()[1] + .85 + 0.8 * i),
                10.,
                0.4,
                fill=True,
                color=c,
                clip_on=False))

        # create text box
        text = "Robertson classification:\n" + \
               "          Type 1\n" + \
               "          Type 2\n" + \
               "          Type 3\n" + \
               "          Type 4\n" + \
               "          Type 5\n" + \
               "          Type 6\n" + \
               "          Type 7\n" + \
               "          Type 8\n" + \
               "          Type 9\n"
        ax3.annotate(text, xy=(1, 1),
                     xycoords='axes fraction',
                     xytext=(20, 0),
                     fontsize=10,
                     textcoords='offset pixels',
                     horizontalalignment='left',
                     verticalalignment='top')

        # save the figure
        plt.savefig(os.path.join(self.output_folder, self.name) + "_lithology.png")
        plt.close()

        # # plot in the robertson chart all the points
        # import robertson
        # classification = robertson.Robertson()
        # classification.soil_types()
        # plt.figure(num=1, figsize=(9, 6), dpi=80)
        # plt.axes().set_position([0.11, 0.11, 0.85, 0.85])
        # plt.plot(classification.soil_type_1[:, 0], classification.soil_type_1[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_2[:, 0], classification.soil_type_2[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_3[:, 0], classification.soil_type_3[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_4[:, 0], classification.soil_type_4[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_5[:, 0], classification.soil_type_5[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_6[:, 0], classification.soil_type_6[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_7[:, 0], classification.soil_type_7[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_8[:, 0], classification.soil_type_8[:, 1], color='k', linewidth=1)
        # plt.plot(classification.soil_type_9[:, 0], classification.soil_type_9[:, 1], color='k', linewidth=1)
        # for i in range(len(self.litho_points)):
        #     plt.plot(self.litho_points[i, 0], self.litho_points[i, 1], marker=".", markersize=4,
        #              color=color_litho[litho[i] - 1], linestyle=None)
        #
        # plt.xlabel("Fr", fontsize=14)
        # plt.ylabel("Qtn", fontsize=14)
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.xlim(0.1, 10)
        # plt.ylim(1, 1000)
        # plt.grid()
        #
        # # save the figure
        # plt.savefig(os.path.join(output_f, self.name) + "_Robertson.png")
        # plt.savefig(os.path.join(output_f, self.name) + "_Robertson.eps")
        # plt.close()

        return

    def write_csv(self):
        """
        Write CSV file into output file.

        Parameters
        ----------
        :param output_f: output folder
        """

        import os

        # write csv
        with open(os.path.join(self.output_folder, str(self.name) + ".csv"), "w") as fo:
            fo.write("Depth NAP [m];Depth [m];tip [kPa];friction [kPa];friction number [-];lithology [-];gamma [kN/m3];"
                     "total stress [-kPa];effective stress [kPa];Qtn [-];Fr [-];IC [-];vs [m/s];G0 [kPa];"
                     "Poisson [-];Damping [%]\n")

            for i in range(len(self.NAP)):
                fo.write(str(self.NAP[i]) + ";" +
                         str(self.depth[i]) + ";" +
                         str(self.tip[i]) + ";" +
                         str(self.friction[i]) + ";" +
                         str(self.friction_nbr[i]) + ";" +
                         str(self.lithology[i]) + ";" +
                         str(self.gamma[i]) + ";" +
                         str(self.total_stress[i]) + ";" +
                         str(self.effective_stress[i]) + ";" +
                         str(self.Qtn[i]) + ";" +
                         str(self.Fr[i]) + ";" +
                         str(self.IC[i]) + ";" +
                         str(self.vs[i]) + ";" +
                         str(self.G0[i]) + ";" +
                         str(self.poisson[i]) + ";" +
                         str(self.damping[i]) + '\n')
        return

def n_iter(n, qt, friction_nb, sigma_eff, sigma_tot, Pa):
    """
    Computation of stress exponent *n*

    :param n: initial stress exponent
    :param qt: tip resistance
    :param friction_nb: friction number
    :param sigma_eff: effective stress
    :param sigma_tot: total stress
    :param Pa: atmospheric pressure
    :return: updated n - stress exponent
    """
    # convergence of n
    import numpy as np 		

    Cn = (Pa / np.array(sigma_eff)) ** n

    Q = ((np.array(qt) - np.array(sigma_tot)) / Pa) * Cn
    F = (np.array(friction_nb) / (np.array(qt) - np.array(sigma_tot))) * 100

    # Q and F cannot be negative. if negative, log10 will be infinite.
    # These values are limited by the contours of soil behaviour of Robertson
    Q[Q <= 1.] = 1.
    F[F <= 0.1] = 0.1
    Q[Q >= 1000.] = 1000.
    F[F >= 10.] = 10.

    IC = ((3.47 - np.log10(Q)) ** 2. + (np.log10(F) + 1.22) ** 2.) ** 0.5

    n = 0.381 * IC + 0.05 * (sigma_eff / Pa) - 0.15
    n[n > 1.] = 1.
    return n


def merge(min_thick, depth, lithology):
    """
    Merge of the CPT layers into a minimum thickness

    :param min_thick: minimum layer thickness
    :param depth: CPT depth
    :param lithology: layer lithology
    :return: thickness, depth, lithology, indices
    """
    import numpy as np 		

    # find location of the start of the soil types
    aux = ""
    idx = []
    # here the i is where I am in the Cpt , val is the value of the lithology which is a number coresponding to Robertson
    # if the previous layer is not the same as the next save the position and save the value for the next step
    for i, val in enumerate(lithology):
        if val != aux:
            aux = val
            idx.append(i)
    # retrieve the depths of the indexes found
    # z_ini - depth at the start of new layer
    z_ini = [depth[i] for i in idx]
    # thickness between those layers calculated
    thick = np.diff(z_ini)
    # soil type
    label = [lithology[i] for i in idx]

    # merge the depths
    try:
        id = np.where(np.array(thick) <= min_thick)[0][0]
    except IndexError:
        return thick, z_ini, label

    del z_ini[id + 1]
    # determine the biggest label
    if thick[id] >= thick[id + 1]:
        del label[id + 1]
    else:
        del label[id]

    thick = np.diff(z_ini)
    return thick, z_ini, label


def compute_probability(coord_cpt, coord_src, coord_rec):
    r"""
    Compute the probability for each scenario following the following formula:

    .. math::

        w_{i} = \frac{1 - \frac{R_{S,i} \cdot \left(R_{S,i} + R_{R,i} \right)}
                               {\sum_{i=1}^{n}R_{S,i} \cdot \left(R_{S,i} + R_{R,i} \right)}}{n - 1}


    where :math:`w_i` is the probability associated with the *i* scenario, :math:`R_{S,i}` the distance between the CPT
    and the source location, :math:`R_{R,i}` the distance between the CPT and the receiver location, and *n* is the
    number of scenarios/CPTs.

    :param coord_cpt: list of coordinates of the CPTs
    :param coord_src: coordinates of the source
    :param coord_rec: coordinates of the receiver
    :return: probs: probability of occurence of the scenario
    """
    import numpy as np

    # number of scenarios
    nb_scenarios = len(coord_cpt)

    distance_to_source = []
    distance_to_receiver = []

    # if there if only one scenario: prob = 100
    if nb_scenarios == 1:
        return [100.]

    # iterate around json
    for i in range(nb_scenarios):
        distance_to_source.append(np.sqrt((coord_cpt[i][0] - coord_src[0])**2 +
                                          (coord_cpt[i][1] - coord_src[1])**2))
        distance_to_receiver.append(np.sqrt((coord_cpt[i][0] - coord_rec[0])**2 +
                                            (coord_cpt[i][1] - coord_rec[1])**2))

    sum_weights = np.sum([distance_to_source[i] * (distance_to_source[i] + distance_to_receiver[i]) for i in range(nb_scenarios)])

    probs = []
    for i in range(nb_scenarios):
        weight = (1 - (distance_to_source[i] * (distance_to_source[i] + distance_to_receiver[i])) / sum_weights) / (nb_scenarios - 1)
        probs.append(weight * 100.)

    return probs
