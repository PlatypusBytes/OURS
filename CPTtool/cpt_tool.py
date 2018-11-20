def read_key(file_name):
    """
    Read CPT key

    Reads key file for the CPT.
    Assumes GEF file convention.

    Parameters
    ----------
    :param file_name: complete path and filename for the key file
    :return: key: Dictionary with the key for CPT interpretation
    """

    # read the key of the GEF files
    with open(file_name, 'r') as f:
        file = f.readlines()

    labels = [line.strip('\n\r').split('=')[0] for line in file]
    dat = [int(line.strip('\n\r').split('=')[1]) for line in file]

    key = {}
    for k in range(len(labels)):
        key.update({labels[k].rstrip(): dat[k]})
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
    import read_gef

    cpts = os.listdir(folder_path)
    cpts = [i for i in cpts if i.upper().endswith(".GEF")]

    jsn = {}
    i = 1
    for f in cpts:
        print("analysis started: " + f)
        test1 = read_gef.CPT()
        test1.read_gef(os.path.join(folder_path, f), key_cpt)
        test1.lithology_calc(gamma_max, pwp_level)
        test1.gamma_calc(gamma_max)
        test1.rho_calc()
        test1.stress_calc(pwp_level)
        # test1.qc1n_calc()
        test1.IC_calc()
        test1.vs_calc()
        test1.damp_calc()
        test1.poisson_calc()
        # test1.merge_thickness(D_min)
        test1.add_json(jsn, i)
        # test1.write_csv(output_folder)
        if make_plots:
            test1.plot_cpt(output_folder)
            test1.plot_lithology(output_folder)
        i += 1
    test1.dump_json(jsn, output_folder)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', help='location of the key file', required=True)
    parser.add_argument('-c', '--cpt', help='location of the cpt folder', required=True)
    parser.add_argument('-o', '--output', help='location of the output folder', required=True)
    parser.add_argument('-t', '--thickness', help='minimum thickness', required=True)
    parser.add_argument('-p', '--plots', help='make plots', required=False, default=False)
    args = parser.parse_args()

    key = read_key(args.key)
    read_cpt(args.cpt, key, args.output, args.thickness, args.plots)
