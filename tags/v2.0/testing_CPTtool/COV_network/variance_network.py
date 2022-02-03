import os
import json
import numpy as np
import matplotlib.pylab as plt


def compute_COV(input_f, output_f):

    # check if output exists
    if not os.path.isdir(output_f):
        os.makedirs(output_f)

    # read all folders results in folder
    folders = os.listdir(input_f)

    # keys
    keys = ["var_E", "var_v", "var_damping", "var_rho", "var_depth"]
    label = {"var_E": "COV E [-]",
             "var_v": "COV Poisson [-]",
             "var_damping": "COV damping [-]",
             "var_rho": "COV Density [-]",
             "var_depth": "COV depth [-]"}

    # create the results dict
    results = dict(var_E=[], var_v=[], var_damping=[], var_rho=[], var_depth=[])
    results_sce = {}

    # for each file
    for f in folders:
        results_sce.update({f: dict(var_E=[], var_v=[], var_damping=[], var_rho=[], var_depth=[])})
        # open json
        try:
            with open(os.path.join(os.path.join(input_f, f), "results_0.json"), "r") as fi:
                data = json.load(fi)
        except FileNotFoundError:
            continue

        for sce in data["scenarios"]:
            for k in keys:
                results[k].extend(sce["data"][k])
                results_sce[f][k].extend(sce["data"][k])

    # create plots
    for k in keys:
        # figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.set_position([0.15, 0.11, 0.8, 0.8])

        ax.plot(range(len(results[k])), results[k], marker="x", linewidth=0)
        ax.set_xlabel("Number [-]")
        ax.set_ylabel(label[k])
        ax.grid()
        ax.set_ylim(0, 5)
        plt.savefig(os.path.join(output_f, k))
        plt.close()

    # create 2nd plots
    for k in keys:
        # figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.set_position([0.15, 0.11, 0.8, 0.8])

        var = []
        for res in results_sce:
            var.append(np.sqrt(np.sum(np.array(results_sce[res][k])**2)))

        ax.plot(range(len(results_sce)), var, marker="x", linewidth=0)
        ax.set_xlabel("Number [-]")
        ax.set_ylabel(label[k])
        ax.grid()
        ax.set_ylim(0, 5)
        plt.savefig(os.path.join(output_f, "point_" + k))
        plt.close()

    return


if __name__ == "__main__":
    compute_COV("../network_analysis/results", "./")
