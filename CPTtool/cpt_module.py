class CPT:
    r"""
    CPT module

    Read and process cpt files: GEF format

    """

    def __init__(self):
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

        # fixed values
        self.g = 9.81
        self.Pa = 100.
        self.a = 0.8

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
        import numpy.core.multiarray as multiarray
        import re

        with open(gef_file, 'r', encoding="Ansi") as f:
            data = f.readlines()

        # search NAP
        idx_nap = [i for i, val in enumerate(data) if val.startswith(r'#ZID=')][0]
        NAP = float(data[idx_nap].split(',')[1])
        # search end of header
        idx_EOH = [i for i, val in enumerate(data) if val.startswith(r'#EOH=')][0]
        # search for coordinates
        idx_coord = [i for i, val in enumerate(data) if val.startswith(r'#XYID=')][0]
        # search index depth
        idx_depth = [int(val.split(',')[0][-1]) - 1 for i, val in enumerate(data)
                     if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['depth'])][0]
        # search index tip resistance
        idx_tip = [int(val.split(',')[0][-1]) - 1 for i, val in enumerate(data)
                   if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['tip'])][0]
        # search index friction
        idx_friction = [int(val.split(',')[0][-1]) - 1 for i, val in enumerate(data)
                        if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['friction'])][0]
        # search index friction number
        idx_friction_nb = [int(val.split(',')[0][-1]) - 1 for i, val in enumerate(data) if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['friction_nb'])][0]

        # search index water if water sensor is available:
        if "water" in key_cpt.keys():
            idx_water = [int(val.split(',')[0][-1]) - 1 for i, val in enumerate(data) if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt['water'])][0]

        # rewrite data with separator ;
        data[idx_EOH + 1:] = [re.sub("[ ,!]", ";", i) for i in data[idx_EOH+1:]]

        # if first column is -9999 it is error and is is neglected
        if float(data[idx_EOH + 1].split(";")[1]) == -9999:
            idx_EOH += 1

        # read data & correct depth to NAP
        depth_tmp = [float(data[i].split(";")[idx_depth]) for i in range(idx_EOH + 1, len(data))]
        tip_tmp = [float(data[i].split(";")[idx_tip]) * 1000. for i in range(idx_EOH + 1, len(data))]
        friction_tmp = [float(data[i].split(";")[idx_friction]) * 1000. for i in range(idx_EOH + 1, len(data))]
        friction_nb_tmp = [float(data[i].split(";")[idx_friction_nb]) for i in range(idx_EOH + 1, len(data))]
        water_tmp = []
        if "water" in key_cpt.keys():
            water_tmp = [float(data[i].split(";")[idx_water]) for i in range(idx_EOH + 1, len(data))]

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
        self.name = os.path.split(gef_file)[-1].upper().split(".GEF")[0]
        self.coord = data[idx_coord].split(',')[1:3]

        self.depth = np.array([np.abs(i) for j, i in enumerate(depth_tmp) if j not in idx_remove])
        self.depth -= self.depth[0]
        self.NAP = np.array([-i + NAP for j, i in enumerate(depth_tmp) if j not in idx_remove])
        self.tip = np.array([i for j, i in enumerate(tip_tmp) if j not in idx_remove])
        self.friction = np.array([i for j, i in enumerate(friction_tmp) if j not in idx_remove])
        self.friction_nbr = np.array([i for j, i in enumerate(friction_nb_tmp) if j not in idx_remove])
        if "water" in key_cpt.keys():
            self.water = np.array([i for j, i in enumerate(water_tmp) if j not in idx_remove])

        return

    def lithology_calc(self, gamma_limit, z_pwp, iter_max=100):
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
        import numpy.core.multiarray as multiarray

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

    def gamma_calc(self, gamma_limit):
        r"""
        Computes unit weight.

        Computes the unit weight following Robertson and Cabal [1]_.
        If unit weight is infinity, it is set to gamma_limit.
        The formula for unit weight is:

        .. math::

            \gamma = 0.27 \log(R_{f}) + 0.36 \log\left(\frac{q_{t}}{Pa}\right) + 1.236

        Parameters
        ----------
        :param gamma_limit: Maximum value for gamma

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 36.
        """
        import numpy as np 		
        import numpy.core.multiarray as multiarray

        # calculate unit weight according to Robertson & Cabal 2015
        np.seterr(divide="ignore")
        aux = 0.27 * np.log10(self.friction_nbr) + 0.36 * np.log10(self.qt / self.Pa) + 1.236
        aux[np.abs(aux) == np.inf] = gamma_limit / 9.81
        self.gamma = aux * 9.81
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
        import numpy.core.multiarray as multiarray

        # compute depth diff
        z = np.diff(np.abs((self.depth - self.depth[0])))
        z = np.append(z, z[-1])
        # total stress
        self.total_stress = np.cumsum(self.gamma * z) + self.depth[0] * np.mean(self.gamma[:10])
        # compute pwp
        # determine location of phreatic line: it cannot be above the CPT depth
        z_aux = np.min([z_pwp, self.NAP[0] + self.depth[0]])
        pwp = (z_aux - self.NAP) * self.g
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
        import numpy.core.multiarray as multiarray

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
        import numpy.core.multiarray as multiarray
        # compute IC
        self.IC = ((3.47 - np.log10(self.Qtn)) ** 2. + (np.log10(self.Fr) + 1.22) ** 2.) ** 0.5
        return

    def vs_calc(self):
        r"""
        Shear wave velocity and shear modulus, following Robertson and Cabal [1]_.

        .. math::

            v_{s} = \left( \alpha_{vs} \cdot \frac{q_{t} - \sigma_{v0}}{Pa} \right)^{0.5}

            \alpha_{vs} = 10^{0.55 I_{c} + 1.68}

            G_{0} = \frac{\gamma}{g} \cdot v_{s}^{2}

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 48-50.
        """

        # vs: following Robertson and Cabal (2015)
        alpha_vs = 10 ** (0.55 * self.IC + 1.68)
        self.vs = (alpha_vs * self.Qtn)**0.5
        self.G0 = self.rho * self.vs**2
        return

    def damp_calc(self):
        r"""
        Damping calculation.

        Damping is assumed as the minimum damping following Darendeli [3]_.

        .. math::

            D_{min} = \left(0.8005 + 0.0129 \cdot PI \cdot OCR^{-0.1069} \right) \cdot \sigma_{v0}'^{-0.2889} \cdot \left[ 1 + -0.0057 \ln \left( freq \right) \right]

        .. rubric:: References
        .. [3] Darendeli, M.B. *Development of a New Family of Normalized Modulus Reduction and material damping curves.* PhD thesis, 2001, pg: 221.
        """
        import numpy as np 		
        import numpy.core.multiarray as multiarray

        # ToDo - define damping
        # assign size to damping
        self.damping = np.zeros(len(self.lithology))
        return

    def poisson_calc(self):
        r"""
        Poisson ratio. Following [2]_.

        Poisson assumed 0.5 for soft layers and 0.2 for sandy layers.

        .. rubric:: References
        .. [2] Mayne, P. *Cone Penetration Testing. A Synthesis of Highway Practice.* Transportation Research Board, 2007, pg: 31.
        """
        import numpy as np 		
        import numpy.core.multiarray as multiarray

        # assign size to poisson
        self.poisson = np.zeros(len(self.lithology))

        for i, lit in enumerate(self.lithology):
            # if soft layer
            if lit == "1" or lit == "2" or lit == "3":
                self.poisson[i] = 0.5
            else:
                self.poisson[i] = 0.2
        return

    def qt_calc(self, litho):
        r"""
        Corrected cone resistance, following Robertson and Cabal [1]_.

        The corrected cone resistance qt is the same as qc for sand, and corrected for the pore water pressure for soft soils.

        .. math::

            q_{t} = q_{c} + u_{2} \left( 1 - a\right)

        Parameters
        ----------
        :param litho: CPT lithology

        .. rubric:: References
        .. [1] Robertson, P.K. and Cabal, K.L. *Guide to Cone Penetration Testing for Geotechnical Engineering.* 6th Edition, Gregg, 2014, pg: 22.
        """

        # qt computed following Robertson & Cabal (2015)
        # qt = qc if sand
        # qt = qc + u2 * (1 - a) if else
        import numpy as np 		
        import numpy.core.multiarray as multiarray

        self.qt = np.zeros(len(litho))

        for i, typ in enumerate(litho):
            # sand
            if int(typ) >= 5:
                self.qt[i] = self.tip[i]
            # not sand
            else:
                self.qt[i] = self.tip[i] + self.water[i] * (1. - self.a)

        return

    def merge_thickness(self, min_layer_thick):
        """
        Reorganises the lithology based on the minimum layer thickness.

        Parameters
        ----------
        :param min_layer_thick: Minimum layer thickness
        """
        import numpy as np 		
        import numpy.core.multiarray as multiarray

        z_ini = self.depth
        label = self.lithology

        a = False
        i = 0
        while not a:
            thick, z_ini, label = merge(float(min_layer_thick), z_ini, label)
            if np.min(thick) >= float(min_layer_thick):
                break
            i += 1
        self.lithology_json = label
        self.depth_json = z_ini
        idx = [int(np.where(self.depth == i)[0][0]) for i in z_ini]
        idx.append(len(self.depth))
        self.indx_json = idx

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
        import numpy.core.multiarray as multiarray

        # create data
        data = {"Lithology": [],
                "Depth": [],
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
            data["Lithology"].append(str(self.lithology_json[i]))
            data["Depth"].append(self.depth_json[i])
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
            damp = 0.05

            data["damping"].append(np.mean(damp))
            data["var_damping"].append(np.std(damp))

        jsn.update({"scenario " + str(id): {}})
        jsn["scenario " + str(id)].update({"Probability": "0.33",
                                           "data": data})

        return

    @staticmethod
    def dump_json(jsn, output_f):
        """
        Dump json file into output file.

        Parameters
        ----------
        :param jsn: json file with data structure
        :param output_f: output folder
        :return:
        """
        import os
        import json

        # if output does no exist -> create
        if not os.path.isdir(output_f):
            os.makedirs(output_f)

        # write file
        with open(os.path.join(output_f, "results.json"), "w") as fo:
            json.dump(jsn, fo, indent=4)
        return

    def plot_cpt(self, output_f, nb_plots=6):
        """
        Plot CPT values.

        Parameters
        ----------
        :param output_f: output folder
        :param nb_plots: (optional) number of plots
        """
        import os
        import numpy as np 		
        import numpy.core.multiarray as multiarray
        import matplotlib.pylab as plt
        from cycler import cycler

        # checks if file_path exits. If not creates file_path
        if not os.path.exists(output_f):
            os.makedirs(output_f)

        # data
        x_data = [self.tip, self.friction_nbr, self.rho, self.G0, self.poisson, self.damping]
        y_data = self.NAP
        l_name = ["Tip resistance", "Friction number", "Density", "Shear modulus", "Poisson ratio", "Damping"]
        x_label = ["Tip resistance [kPa]", "Friction number [-]", r"Density [kg/m$^{3}$]", "Shear modulus [kPa]", "Poisson ratio [-]", "Damping [%]"]
        y_label = "Depth NAP [m]"

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

        plt.tight_layout()
        # save the figure
        plt.savefig(os.path.join(output_f, self.name) + "_cpt.png")
        plt.close()

        return

    def plot_lithology(self, output_f):
        """
        Plot CPT lithology.

        Parameters
        ----------
        :param output_f: output folder
        """
        import os
        import numpy as np 		
        import numpy.core.multiarray as multiarray
        import matplotlib.pylab as plt
        import matplotlib.patches as patches

        # checks if file_path exits. If not creates file_path
        if not os.path.exists(output_f):
            os.makedirs(output_f)

        # define figure
        f, (ax1, ax3) = plt.subplots(1, 2, figsize=(7, 10), sharey=True)

        # first subplot - CPT
        # ax1 tip
        ax1.set_position([0.15, 0.1, 0.4, 0.8])
        ax1.plot(self.tip, self.NAP, label="Tip resistance", color="b")
        ax1.set_xlabel("Tip resistance [kPa]", fontsize=12)
        ax1.set_ylabel("Depth NAP [m]", fontsize=12)
        ax1.set_xlim(left=0)

        # ax2 friction number
        ax2 = ax1.twiny()
        ax2.set_position([0.15, 0.1, 0.4, 0.8])
        ax2.plot(self.friction_nbr, self.NAP, label="Friction number", color="r")
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
        diff_nap = np.diff(self.NAP)

        # color the field
        for i in range(len(self.NAP) - 1):
            ax3.add_patch(patches.Rectangle(
                (0, self.NAP[i]),
                10.,
                diff_nap[i],
                fill=True,
                color=color_litho[litho[i] - 1]))

        # create legend for Robertson
        for i, c in enumerate(color_litho):
            ax3.add_patch(patches.Rectangle(
                (16, ax1.get_ylim()[1]-.85 - 0.565 * i),
                10.,
                0.2,
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
        plt.savefig(os.path.join(output_f, self.name) + "_lithology.png")
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
    import numpy.core.multiarray as multiarray

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
    import numpy.core.multiarray as multiarray

    # find location of the start of the soil types
    aux = ""
    idx = []
    for i, val in enumerate(lithology):
        if val != aux:
            aux = val
            idx.append(i)

    # z_ini - depth at the start of new layer
    z_ini = [depth[i] for i in idx]
    # thickness
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
