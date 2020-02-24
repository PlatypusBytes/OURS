import os
import numpy as np
import robertson
import tools_utils
import matplotlib.pylab as plt
import matplotlib.patches as patches
from cycler import cycler


class CPT:
    r"""
    CPT module

    Read and process cpt files: GEF format
    """

    def __init__(self, out_fold):
        # variables
        self.depth = []
        self.coord = []
        self.NAP = []
        self.tip = []
        self.friction = []
        self.friction_nbr = []
        self.a = []
        self.name = []
        self.gamma = []
        self.rho = []
        self.total_stress = []
        self.effective_stress = []
        self.pwp = []
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

        return

    def parse_bro(self, cpt, minimum_length=5, minimum_samples=50, minimum_ratio=0.1):
        """
        Parse the BRO information into the object structure

        :param cpt: BRO cpt dataset
        :param minimum_length: minimum length that cpt files needs to have
        :param minimum_samples: minimum samples that cpt files needs to have
        :param minimum_ratio: minimum ratio of positive values that cpt files needs to have
        :return:
        """

        # remove NAN row from the dataframe
        for key in cpt["dataframe"]:
            cpt["dataframe"] = cpt["dataframe"].dropna(subset=[key])

        # check if file contains data
        if len(cpt["dataframe"].penetrationLength) == 0:
            message = "File " + cpt["id"] + " contains no data"
            return message

        # check if data is different than zero:
        keys = ['penetrationLength', 'coneResistance', 'localFriction', 'frictionRatio']
        for k in keys:
            if all(cpt["dataframe"][k] == 0):
                message = "File " + cpt["id"] + " contains empty data"
                return message

        # parse cpt file name
        self.name = cpt['id']
        # parse coordinates
        self.coord = [cpt['location_x'], cpt['location_y']]

        # check criteria of minimum length
        if np.max(np.abs(cpt['dataframe'].penetrationLength.values)) < minimum_length:
            message = "File " + cpt["id"] + " has a length smaller than " + str(minimum_length)
            return message

        # check criteria of minimum samples
        if len(cpt['dataframe'].penetrationLength.values) < minimum_samples:
            message = "File " + cpt["id"] + " has a number of samples smaller than " + str(minimum_samples)
            return message

        # check data consistency: remove doubles depth
        cpt["dataframe"] = cpt["dataframe"].drop_duplicates(subset='penetrationLength', keep="first")

        # check if there is a pre_drill. if so pad the data
        depth, cone_resistance, friction_ratio, local_friction, pore_pressure = self.define_pre_drill(cpt,
                                                                                                      length_of_average_points=minimum_samples)

        # check quality of CPT
        # if more than minimum_ratio CPT is corrupted: discard CPT
        if (
                len(cone_resistance[cone_resistance <= 0]) / len(cone_resistance) > minimum_ratio
                or len(cone_resistance[local_friction <= 0]) / len(local_friction) > minimum_ratio
            ):
            message = "File " + cpt["id"] + " is corrupted"
            return message

        # parse depth
        self.depth = depth
        # parse NAP depth
        self.NAP = cpt['offset_z'] - depth
        # parse tip resistance
        self.tip = cone_resistance * 1000.
        self.tip[self.tip <= 0] = 0.
        # parse friction
        self.friction = local_friction * 1000.
        self.friction[self.friction <= 0] = 0.
        # parser friction number
        self.friction_nbr = friction_ratio
        self.friction_nbr[self.friction_nbr <= 0] = 0.
        # read a
        self.a = cpt['a']
        # default water is zero
        self.water = np.zeros(len(self.depth))
        # if water exists parse water
        if "porePressureU2" in cpt["dataframe"]:
            self.water = pore_pressure * 1000.
        return True

    def smooth(self, nb_points=5):
        r"""
        Smooth the cpt input data

        :param nb_points: (optional) number of points for smoothing. default 5
        :return:
        """

        self.tip = tools_utils.smooth(self.tip, window_len=nb_points, lim=0)
        self.friction = tools_utils.smooth(self.friction, window_len=nb_points, lim=0)
        self.friction_nbr = tools_utils.smooth(self.friction_nbr, window_len=nb_points, lim=0)
        self.water = tools_utils.smooth(self.water, window_len=nb_points, lim=0)
        return

    def define_pre_drill(self, cpt_BRO, length_of_average_points=3):
        r"""
        Checks the existence of pre-drill.
        If predrill exists it add the average value of tip, friction and friction number to the pre-drill length.
        The average is computed over the length_of_average_points.
        If pore water pressure is measured, the pwp is assumed to be zero at surface level.

        :param cpt_BRO: BRO cpt dataset
        :param length_of_average_points: number of samples of the CPT to be used to fill pre-drill
        :return: depth, tip resistance, friction number, friction, pore water pressure
        """

        starting_depth = 0
        pore_pressure = None
        if float(cpt_BRO['predrilled_z']) != 0.:
            # if there is pre-dill add the average values to the pre-dill

            # Set the discretisation
            dicretisation = np.average(np.diff(cpt_BRO['dataframe']['penetrationLength'].values))

            # find the average
            average_cone_res = np.average(cpt_BRO['dataframe']['coneResistance'][:length_of_average_points])
            average_fr_ratio = np.average(cpt_BRO['dataframe']['frictionRatio'][:length_of_average_points])
            average_loc_fr = np.average(cpt_BRO['dataframe']['localFriction'][:length_of_average_points])

            # Define all in the lists
            local_depth = np.arange(starting_depth, float(cpt_BRO['predrilled_z']), dicretisation)
            local_cone_res = np.repeat(average_cone_res, len(local_depth))
            local_fr_ratio = np.repeat(average_fr_ratio, len(local_depth))
            local_loc_fr = np.repeat(average_loc_fr, len(local_depth))

            # if there is pore water pressure
            # Here the endpoint is False so that for the final of local_pore_pressure I don't end up with the same value
            # as the first in the Pore Pressure array.
            if "porePressureU2" in cpt_BRO["dataframe"]:
                local_pore_pressure = np.linspace(0, cpt_BRO['dataframe']["porePressureU2"].values[0], len(local_depth),
                                                  endpoint=False)
                pore_pressure = np.append(local_pore_pressure, cpt_BRO['dataframe']["porePressureU2"].values)

            # Enrich the Penetration Length
            depth = np.append(local_depth, local_depth[-1] + dicretisation + cpt_BRO['dataframe']['penetrationLength'].values - cpt_BRO['dataframe']['penetrationLength'].values[0])
            coneresistance = np.append(local_cone_res, cpt_BRO['dataframe']['coneResistance'].values)
            frictionratio = np.append(local_fr_ratio, cpt_BRO['dataframe']['frictionRatio'].values)
            localfriction = np.append(local_loc_fr, cpt_BRO['dataframe']['localFriction'].values)

        else:
            # No predrill existing: just parsing data
            depth = cpt_BRO['dataframe']['penetrationLength'].values - cpt_BRO['dataframe']['penetrationLength'].values[0]
            coneresistance = cpt_BRO['dataframe']['coneResistance'].values
            frictionratio = cpt_BRO['dataframe']['frictionRatio'].values
            localfriction = cpt_BRO['dataframe']['localFriction'].values

            # if there is pore water pressure
            if "porePressureU2" in cpt_BRO['dataframe']:
                pore_pressure = cpt_BRO['dataframe']['porePressureU2'].values

        # correct for missing samples in the top of the CPT
        if depth[0] > 0:
            # add zero
            depth = np.append(0, depth)
            coneresistance = np.append(np.average(cpt_BRO['dataframe']['coneResistance'][:length_of_average_points]), coneresistance)
            frictionratio = np.append(np.average(cpt_BRO['dataframe']['frictionRatio'][:length_of_average_points]), frictionratio)
            localfriction = np.append(np.average(cpt_BRO['dataframe']['localFriction'][:length_of_average_points]), localfriction)

            # if there is pore water pressure
            if "porePressureU2" in cpt_BRO["dataframe"]:
                pore_pressure = np.append(np.average(cpt_BRO['dataframe']['porePressureU2'][:length_of_average_points]), pore_pressure)

        return depth, coneresistance, frictionratio, localfriction, pore_pressure

    def lithology_calc(self):
        r"""
        Lithology calculation.

        Computes the lithology following Robertson and Cabal :cite:`robertson_cabal_2014`.
        """

        classification = robertson.Robertson()
        classification.soil_types()

        # compute Qtn and Fr
        self.norm_calc()

        litho, points = classification.lithology(self.Qtn, self.Fr)

        # assign to variables
        self.lithology = litho
        self.litho_points = points

        return

    def pwp_level_calc(self, path_bro):
        """
        Computes the estimated pwp level for the cpt coordinate

        :param path_bro: path for the location of the netCDF file with expected water levels
        :return:
        """
        import netcdf
        pwp = netcdf.NetCDF()
        pwp.read_cdffile(path_bro)
        pwp.query(self.coord[0], self.coord[1])
        self.pwp = pwp.NAP_water_level

        return

    def gamma_calc(self, gamma_limit, method="Robertson", gamma_min=10):
        r"""
        Computes unit weight.

        Computes the unit weight following Robertson and Cabal :cite:`robertson_cabal_2014`.
        If unit weight is infinity, it is set to gamma_limit.
        The formula for unit weight is:

        .. math::

            \gamma = 0.27 \log(R_{f}) + 0.36 \log\left(\frac{q_{t}}{Pa}\right) + 1.236

        Alternative method of Lengkeek et al. :cite:`lengkeek_2018`:

        .. math::

            \gamma = \gamma_{sat,ref} - \beta
            \left( \frac{\log \left( \frac{q_{t,ref}}{q_{t}} \right)}{\log \left(\frac{R_{f,ref}}{R_{f}}\right)} \right)

        Parameters
        ----------
        :param gamma_limit: Maximum value for gamma
        :param method: (optional) Method to compute unit weight. Default is Robertson
        :param gamma_min: (optional) Minimum gamma. Default is 10
        """

        np.seterr(divide="ignore", over='print')

        # calculate unit weight according to Robertson & Cabal 2015
        if method == "Robertson":
            aux = 0.27 * np.log10(self.friction_nbr) + 0.36 * np.log10(self.qt / self.Pa) + 1.236
            aux[np.abs(aux) == np.inf] = gamma_limit / self.g
            aux = tools_utils.ceil_value(aux, gamma_min / self.g)
            self.gamma = aux * self.g

        elif method == "Lengkeek":
            aux = 19. - 4.12 * np.log10(5000. / self.qt) / np.log10(30. / self.friction_nbr)
            aux[np.abs(aux) == np.inf] = gamma_limit  # should this be divided with self.g ?
            aux = tools_utils.ceil_value(aux, gamma_min)
            self.gamma = aux

        elif method == "all":  # if all, compares all the methods and plot
            self.gamma_calc(gamma_limit, method="Lengkeek")
            gamma_1 = self.gamma
            self.gamma_calc(gamma_limit, method="Robertson")
            gamma_2 = self.gamma
            self.plot_correlations([gamma_1, gamma_2], "Unit Weight [kN/m3]", ["Lengkeek", "Robertson"], "unit_weight")
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

    def stress_calc(self):
        r"""
        Computes total and effective stress
        """

        # compute depth diff
        z = np.diff(np.abs((self.depth - self.depth[0])))
        z = np.append(z, z[-1])
        # total stress
        self.total_stress = np.cumsum(self.gamma * z) + self.depth[0] * np.mean(self.gamma[:10])
        # compute pwp
        # determine location of phreatic line: it cannot be above the CPT depth
        z_aux = np.min([self.pwp, self.NAP[0] + self.depth[0]])
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
        normalisation of qc and friction into Qtn and Fr, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            Q_{tn} = \left(\frac{q_{t} - \sigma_{v0}}{Pa} \right) \left(\frac{Pa}{\sigma_{v0}'}\right)^{n}

            F_{r} = \frac{f_{s}}{q_{t}-\sigma_{v0}} \cdot 100


        Parameters
        ----------
        :param n_method: (optional) parameter *n* stress exponent. Default is n computed in an iterative way.
        """

        # normalisation of qc and friction into Qtn and Fr: following Robertson and Cabal (2014)

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
                n1 = tools_utils.n_iter(n, self.tip, self.friction_nbr, self.effective_stress, self.total_stress,
                                        self.Pa)
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

    def IC_calc(self):
        r"""
        IC, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            I_{c} = \left[ \left(3.47 - \log\left(Q_{tn}\right) \right)^{2} + \left(\log\left(F_{r}\right) + 1.22 \right)^{2} \right]^{0.5}

        """

        # IC: following Robertson and Cabal (2015)
        # compute IC
        self.IC = ((3.47 - np.log10(self.Qtn)) ** 2. + (np.log10(self.Fr) + 1.22) ** 2.) ** 0.5
        return

    def vs_calc(self, method="Robertson"):
        r"""
        Shear wave velocity and shear modulus. The following methods are available:

        * Robertson and Cabal :cite:`robertson_cabal_2014`:

        .. math::

            v_{s} = \left( \alpha_{vs} \cdot \frac{q_{t} - \sigma_{v0}}{Pa} \right)^{0.5}

            \alpha_{vs} = 10^{0.55 I_{c} + 1.68}

            G_{0} = \frac{\gamma}{g} \cdot v_{s}^{2}

        * Mayne :cite:`mayne_2006`:

        .. math::

            v_{s} = e^{\frac{\gamma_{sat} + 4.03}{4.17}} \cdot \left( \frac{\sigma_{v0}'}{\sigma_{atm}} \right)^{0.25}

            v_{s} = 118.8 \cdot \log \left(f_{s} \right) + 18.5

        * Andrus *et al.* :cite:`andrus_2007`:

        .. math::

            v_{s} = 2.27 \cdot q_{t}^{0.412} \cdot I_{c}^{0.989} \cdot D^{0.033} \cdot ASF  (Holocene)

            v_{s} = 2.62 \cdot q_{t}^{0.395} \cdot I_{c}^{0.912} \cdot D^{0.124} \cdot SF   (Pleistocene)

        * Zhang and Tong :cite:`zhang_2017`:

        .. math::

            v_{s} = 10.915 \cdot q_{t}^{0.317} \cdot I_{c}^{0.210} \cdot D^{0.057} \cdot SF^{a}  (Holocene)

        * Ahmed :cite:`ahmed_2017`:

        .. math::

            v_{s} = 1000 \cdot e^{-0.887 \cdot I_{c}} \cdot \left( \left(1 + 0.443 \cdot F_{r} \right) \cdot \left(\frac{\sigma'_{v}}{p_{a}} \right) \cdot \left(\frac{\gamma_{w}}{\gamma} \right) \right)^{0.5}
        """

        if method == "Robertson":
            # vs: following Robertson and Cabal (2015)
            alpha_vs = 10 ** (0.55 * self.IC + 1.68)
            vs = alpha_vs * (self.qt - self.total_stress) / self.Pa
            vs = tools_utils.ceil_value(vs, 0)
            self.vs = vs ** 0.5
            self.G0 = self.rho * self.vs**2
        elif method == "Mayne":
            # vs: following Mayne (2006)
            vs = np.exp((self.gamma + 4.03) / 4.17) * (self.effective_stress / self.Pa) ** 0.25
            self.vs = tools_utils.ceil_value(vs, 0)
            self.G0 = self.rho * self.vs ** 2
        elif method == "Andrus":
            # vs: following Andrus (2007)
            vs = 2.27 * self.qt ** 0.412 * self.IC ** 0.989 * self.depth ** 0.033 * 1
            self.vs = tools_utils.ceil_value(vs, 0)
            self.G0 = self.rho * self.vs ** 2
        elif method == "Zang":
            # vs: following Zang & Tong (2017)
            vs = 10.915 * self.qt ** 0.317 * self.IC ** 0.210 * self.depth ** 0.057 * 0.92
            self.vs = tools_utils.ceil_value(vs, 0)
            self.G0 = self.rho * self.vs ** 2
        elif method == "Ahmed":
            vs = 1000. * np.exp(-0.887 * self.IC) * (1. + 0.443 * self.Fr * self.effective_stress / self.Pa * self.g
                                                          / self.gamma) ** 0.5
            self.vs = tools_utils.ceil_value(vs, 0)
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
        return

    def damp_calc(self, d_min=2, Cu=2., D50=0.2, Ip=40., method="Mayne", freq=1.):
        r"""
        Damping calculation.

        For clays and peats, the damping is assumed as the minimum damping following Darendeli :cite:`darendeli_2001`.

        .. math::

            D_{min} = \left(0.8005 + 0.0129 \cdot PI \cdot OCR^{-0.1069} \right) \cdot \sigma_{v0}'^{-0.2889} \cdot \left[ 1 + 0.2919 \ln \left( freq \right) \right]

        The OCR can be computed according to Mayne :cite:`mayne_2007` or Robertson and Cabal :cite:`robertson_cabal_2014`.


        .. math::

            OCR_{Mayne} = 0.33 \cdot \frac{q_{t} - \sigma_{v0}}{\sigma_{v0}'}

            OCR_{Rob} = 0.25 \left(Q_{t}\right)^{1.25}


        For sand the damping is assumed as the minimum damping following Menq :cite`menq_2003`.

        .. math::
            D_{min} = 0.55 \cdot C_{u}^{0.1} \cdot d_{50}^{-0.3} \cdot  \left(\frac{\sigma'_{v}}{p_{a}} \right)^-0.08

        Parameters
        ----------
        :param d_min: (optional) Minimum damping. Default is 2%
        :param Cu: (optional) Coefficient of uniformity. Default is 2.0
        :param D50: (optional) Median grain size. Default is 0.2 mm
        :param Ip: (optional) Plasticity index. Default is 40
        :param method: (optional) Method for calculation of OCR. Default is Mayne
        :param freq: (optional) Frequency. Default is 1 Hz
        """

        # assign size to damping
        self.damping = np.zeros(len(self.lithology)) + d_min
        OCR = np.zeros(len(self.lithology))

        for i, lit in enumerate(self.lithology):
            # if  clay
            if lit == "3" or lit == "4" or lit == "5":
                if method == "Mayne":
                    OCR[i] = 0.33 * (self.qt[i] - self.total_stress[i]) / self.effective_stress[i]
                elif method == "Robertson":
                    OCR[i] = 0.25 * self.Qtn[i] ** 1.25
                self.damping[i] = (0.8005 + 0.0129 * Ip * OCR[i] ** (-0.1069)) * \
                                  (self.effective_stress[i] / self.Pa) ** (-0.2889) * (1 + 0.2919 * np.log(freq))
            # if peat
            elif lit == "1" or lit == "2":
                # same as clay: OCR=1 IP=100
                self.damping[i] = 2.512 * (self.effective_stress[i] / self.Pa) ** -0.2889
            # if sand
            else:
                self.damping[i] = 0.55 * Cu ** 0.1 * D50 ** -0.3 * (self.effective_stress[i] / self.Pa) ** -0.08

        # limit the damping (when stress is zero damping is infinite)
        self.damping[self.damping == np.inf] = 100
        # damping units -> dimensionless
        self.damping /= 100
        return

    def poisson_calc(self):
        r"""
        Poisson ratio. Following Mayne :cite:`mayne_2007`.

        Poisson assumed 0.495 for soft layers, 0.2 for silty layers and 0.3 for sandy layers.
        """

        # assign size to poisson
        self.poisson = np.zeros(len(self.lithology))

        for i, lit in enumerate(self.lithology):
            # if soft layer
            if lit == "1" or lit == "2" or lit == "3":
                self.poisson[i] = 0.495
            elif lit == "4" or lit == "5":
                self.poisson[i] = 0.2
            else:
                self.poisson[i] = 0.3
        return

    def qt_calc(self):
        r"""
        Corrected cone resistance, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            q_{t} = q_{c} + u_{2} \left( 1 - a\right)
        """

        # qt computed following Robertson & Cabal (2015)
        # qt = qc + u2 * (1 - a)
        self.qt = self.tip + self.water * (1. - self.a)
        self.qt[self.qt <= 0] = 0
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
        # data
        x_data = [self.tip, self.friction_nbr, self.rho, self.G0, self.poisson, self.damping]
        y_data = self.depth
        l_name = ["Tip resistance", "Friction number", "Density", "Shear modulus", "Poisson ratio", "Damping"]
        x_label = ["Tip resistance [kPa]", "Friction number [-]", r"Density [kg/m$^{3}$]", "Shear modulus [kPa]",
                   "Poisson ratio [-]", "Damping [-]"]
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

        # define figure
        f, (ax1, ax3) = plt.subplots(1, 2, figsize=(7, 10), sharey=True)

        # first subplot - CPT
        # ax1 tip
        ax1.set_position([0.15, 0.1, 0.4, 0.8])
        l1, = ax1.plot(self.tip, self.depth, label="Tip resistance", color="b")
        ax1.set_xlabel("Tip resistance [kPa]", fontsize=12)
        ax1.set_ylabel("Depth [m]", fontsize=12)
        ax1.set_xlim(left=0)
        # invert y axis
        ax1.invert_yaxis()

        # ax2 friction number
        ax2 = ax1.twiny()
        ax2.set_position([0.15, 0.1, 0.4, 0.8])
        l2, = ax2.plot(self.friction_nbr, self.depth, label="Friction number", color="r")
        ax2.set_xlabel("Friction number [-]", fontsize=12)
        ax2.set_xlim(left=0)

        # align grid
        ax1.set_xticks(np.linspace(0, ax1.get_xticks()[-1], 5))
        ax2.set_xticks(np.linspace(0, ax2.get_xticks()[-1], 5))

        # grid
        ax1.grid()

        # legend
        plt.legend(handles=[l1, l2], loc="upper right")

        # second subplot - Lithology
        litho = np.array(self.lithology).astype(int)
        color_litho = ["red", "brown", "cyan", "blue", "gray", "yellow", "orange", "green", "lightgray"]

        ax3.set_position([0.60, 0.1, 0.05, 0.8])
        ax3.set_xlim((0, 10.))
        # remove ticks
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.tick_params(axis='both', which='both', length=0)

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
                (16, ax1.get_ylim()[1] + (ax1.get_ylim()[0] - ax1.get_ylim()[1]) * 0.0425
                 + (ax1.get_ylim()[0] - ax1.get_ylim()[1]) * 0.02 * i),
                10.,
                (ax1.get_ylim()[0] - ax1.get_ylim()[1]) * 0.015,
                fill=True,
                color=c,
                clip_on=False))

        # create text box
        text = "Robertson classification:\n\n" + \
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

        return

    def write_csv(self):
        """
        Write CSV file into output file.
        """

        # write csv
        with open(os.path.join(self.output_folder, str(self.name) + ".csv"), "w") as fo:
            fo.write("Depth NAP [m];Depth [m];tip [kPa];friction [kPa];friction number [-];lithology [-];gamma [kN/m3];"
                     "total stress [-kPa];effective stress [kPa];Qtn [-];Fr [-];IC [-];vs [m/s];G0 [kPa];"
                     "Poisson [-];Damping [-]\n")

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
