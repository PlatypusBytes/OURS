import json
import numpy as np
import matplotlib.pylab as plt
import os
from scipy.stats import lognorm


def read_json(file_name):
    # load json file
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def read_SOS(file_name, segment, min_dept=-20):

    key = [['H_Aa_ht', 8],
           ['H_Vbv_v', 1],
           ['H_Vhv_v', 1],
           ['H_Rk_k&v', 2],
           ['H_Rk_vk', 2],
           ['H_Rk_k', 3],
           ['H_Ro_z&k', 4],
           ['H_Rr_o&z', 5],
           ['H_Rg_zm', 6],
           ['P_Wrd_zm', 6],
           ['P_Rk_k&s', 6],
           ['P_Rg_zm', 6],
        ]

    data = read_json(file_name)

    depth = []
    label = []
    E = []
    rho = []
    for i in range(len(data[segment]['StochasticProfile']["Z_top"])):
        dep = np.diff(list(map(float, data[segment]['StochasticProfile']["Z_top"][i]))).tolist()
        dep.insert(0, 0)
        aux = data[segment]['StochasticProfile']["Name"][i]
        var = []
        for k in aux:
            idx = [j[0] for j in key]
            idx = idx.index(k)
            var.append(key[idx][1])
        dep.append(min_dept)
        var.append(var[-1])
        depth.append(np.cumsum(dep) * -1)
        label.append(var)
        ee = list(map(float, data[segment]['StochasticProfile']["Young"][i]))
        ee.append(ee[-1])
        E.append(ee)
        r = list(map(float, data[segment]['StochasticProfile']["gamma_sat"][i]))
        r.append(r[-1])
        r = [i /9.81 for i in r]
        rho.append(r)

    return depth, label, E, rho


def read_OURS(file_name):
    data = read_json(file_name)

    depth = []
    depth_std = []
    litho = []
    E = []
    E_std = []
    rho = []
    rho_std = []
    for i in range(len(data["scenarios"])):
        depth.append([j for j in data["scenarios"][i]["data"]["depth"]])
        depth_std.append([j for j in data["scenarios"][i]["data"]["var_depth"]])
        E.append([j / 1e6 for j in data["scenarios"][i]["data"]["E"]])
        E_std.append([j for j in data["scenarios"][i]["data"]["var_E"]])
        rho.append([j / 1000 for j in data["scenarios"][i]["data"]["rho"]])
        rho_std.append([j for j in data["scenarios"][i]["data"]["var_rho"]])

        litho_aux = data["scenarios"][i]["data"]["lithology"]
        aux = []
        for j in litho_aux:
            aux.append(np.mean(list(map(float, j.split("/")))))
        litho.append(aux)

    return depth, depth_std, litho, E, E_std, rho, rho_std


def do_fig(depth_SOS, E3, depth_ours, E_ours, cov_E, title, limx, limy, lab):
    plt.figure(figsize=(6, 8))

    # plot the SOS
    for i in range(len(E3)):
        plt.step(E3[i], depth_SOS[i], label="SOS scenario " + str(i + 1), color='k', linewidth=2)

    # plot the interpolation
    for j in range(len(E_ours)):
        plt.fill_betweenx(depth_ours[j],
                          np.array(E_ours[j]) + np.array(E_ours[j]) * np.array(cov_E[j]) * 1.96,
                          np.array(E_ours[j]) - np.array(E_ours[j]) * np.array(cov_E[j]) * 1.96, alpha=0.1,
                          color='b')
        plt.plot(np.array(E_ours[j]), depth_ours[j], label="OURS scenario " + str(j + 1), linewidth=0.5,
                 color='b')
    plt.ylim(limy)
    plt.xlim(limx)
    plt.xlabel(lab)
    plt.ylabel("Depth [m]")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join("./", title))
    plt.close()
    return


d, _, E_sos, rho_sos = read_SOS(r"N:\Projects\1230500\1230936\B. Measurements and calculations\static\results_AF_1.json", "segment_7")
d_ours, _, lit, E, cov_e, rho, cov_rho = read_OURS(r"C:\Users\zuada\software_dev\OURS\Validation\segment_7\results_0.json")
do_fig(d, E_sos, d_ours, E, cov_e, "segment_7_E", [0, 1000], [20, 0], "Young modulus [MPa]")
do_fig(d, rho_sos, d_ours, rho, cov_rho, "segment_7_rho", [0, 2.5], [20, 0], 'Density [t/m$^{3}$]')

d, _, E_sos, rho_sos = read_SOS(r"N:\Projects\1230500\1230936\B. Measurements and calculations\static\results_AF_1.json", "segment_13")
d_ours, _, lit, E, cov_e, rho, cov_rho = read_OURS(r"C:\Users\zuada\software_dev\OURS\Validation\segment_13\results_0.json")
do_fig(d, E_sos, d_ours, E, cov_e, "segment_13_E", [0, 400], [7.5, 0], "Young modulus [MPa]")
do_fig(d, rho_sos, d_ours, rho, cov_rho, "segment_13_rho", [0, 2.5], [7.5, 0], 'Density [t/m$^{3}$]')

d, _, E_sos, rho_sos = read_SOS(r"N:\Projects\1230500\1230936\B. Measurements and calculations\static\results_AF_1.json", "segment_1")
d_ours, _, lit, E, cov_e, rho, cov_rho = read_OURS(r"C:\Users\zuada\software_dev\OURS\Validation\segment_1\results_0.json")
do_fig(d, E_sos, d_ours, E, cov_e, "segment_1_E", [0, 500], [20, 0], "Young modulus [MPa]")
do_fig(d, rho_sos, d_ours, rho, cov_rho, "segment_1_rho", [0, 2.5], [20, 0], 'Density [t/m$^{3}$]')
