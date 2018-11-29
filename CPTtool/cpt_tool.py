def set_key():
    """
    Define CPT key

    Parameters
    ----------
    :return: key: Dictionary with the key for CPT interpretation
    """

    key = {}
    labels = ['depth', 'tip', 'friction', 'friction_nb', 'water']
    dat = [1, 2, 3, 4, 6]
    for k in range(len(labels)):
        key.update({labels[k]: dat[k]})
    return key


def read_cpt(folder_path, key_cpt, output_folder, D_min, make_plots, gamma_max=22, pwp_level=0):
    """
    Read CPT

    Read and process cpt files: GEF format

    Parameters
    ----------
    :param folder_path: Folder where the cpt files are located
    :param key_cpt: File with the key for the CPT interpretation
    :param output_folder: Folder to save the files
    :param D_min: Minimum layers thickness
    :param gamma_max: (optional) maximum value specific weight soil
    :param pwp_level: (optional) pore water level in NAP

    """

    # read the cpt files
    import os
    import cpt_module

    cpts = os.listdir(folder_path)
    cpts = [i for i in cpts if i.upper().endswith(".GEF")]

    jsn = {}
    i = 1
    for f in cpts:
        print("analysis started: " + f)
        cpt = cpt_module.CPT(output_folder)
        cpt.read_gef(os.path.join(folder_path, f), key_cpt)
        cpt.lithology_calc_iter(gamma_max, pwp_level)
        cpt.gamma_calc(gamma_max)
        cpt.rho_calc()
        cpt.stress_calc(pwp_level)
        # cpt.qc1n_calc()
        cpt.IC_calc()
        cpt.vs_calc()
        cpt.damp_calc()
        cpt.poisson_calc()
        # cpt.merge_thickness(D_min)
        cpt.add_json(jsn, i)
        if make_plots:
            cpt.plot_cpt()
            cpt.plot_lithology()
        i += 1
    cpt.dump_json(jsn)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cpt', help='location of the cpt folder', required=True)
    parser.add_argument('-o', '--output', help='location of the output folder', required=True)
    parser.add_argument('-t', '--thickness', help='minimum thickness', required=True)
    parser.add_argument('-p', '--plots', help='make plots', required=False, default=False)
    args = parser.parse_args()

    key = set_key()
    read_cpt(args.cpt, key, args.output, args.thickness, args.plots)
