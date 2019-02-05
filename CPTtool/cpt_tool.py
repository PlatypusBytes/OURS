import argparse
import bro


def define_methods(input_file):
    """
    Defines correlations for the CPT

    :param input_file: json file with methods
    :return: methods: dictionary with the methods for CPT correlations
    """
    import os
    import sys
    import json

    # possible keys:
    keys = ["gamma", "vs", "OCR"]
    # possible key-values
    gamma_keys = ["Robertson", "Lengkeek", "all"]
    vs_keys = ["Robertson", "Mayne", "Andrus", "Zang", "Ahmed", "all"]
    OCR_keys = ["Mayne", "Robertson"]

    # if no file is available: -> uses default values
    if not input_file:
        methods = {"gamma": gamma_keys[0],
                   "vs": vs_keys[0],
                   "OCR": OCR_keys[0]
                   }
        return methods

    # check if input file exists
    if not os.path.isfile(input_file):
        sys.exit("File methods does not exist")

    # if the file is available
    with open(input_file, "r") as f:
        data = json.load(f)

    # # checks if the keys are correct
    for i in data.keys():
        if not any(i in k for k in keys):
            sys.exit("Error: Key " + i + " is not known. Keys must be: " + ', '.join(keys))

    # check if the key-values are correct for gamma
    if not any(data["gamma"] in k for k in gamma_keys):
        sys.exit("Error: gamma key is not known. gamma keys must be: " + ', '.join(gamma_keys))
    # check if the key-values are correct for vs
    if not any(data["vs"] in k for k in vs_keys):
        sys.exit("Error: gamma key is not known. vs keys must be: " + ', '.join(vs_keys))
    # check if the key-values are correct for OCR
    if not any(data["OCR"] in k for k in OCR_keys):
        sys.exit("Error: gamma key is not known. OCR keys must be: " + ', '.join(OCR_keys))

    methods = {"gamma": data["gamma"],
               "vs": data["vs"],
               "OCR": data["OCR"]
               }
    return methods


def read_json(input_file):
    """
    Reads input json file

    :param input_file: json file with the input values
    :return: data: dictionary with the input files
    """
    import os
    import sys
    import json

    # check if file exits
    if not os.path.isfile(input_file):
        sys.exit("Input JSON file does not exist")

    # read file
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


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


def read_cpt(cpt_BRO, methods, output_folder, input_dictionary, make_plots, index, gamma_max=22, pwp_level=0):
    """
    Read CPT

    Read and process cpt files: GEF format

    Parameters
    ----------
    :param cpt_BRO: cpt information from the BRO
    :param key_cpt: File with the key for the CPT interpretation
    :param methods: Methods for the CPT correlations
    :param output_folder: Folder to save the files
    :param input_dictionary: Dictionary with input settings
    :param make_plots: Bool to make plots
    :param index: index of the calculation point
    :param input_dictionary: Dictionary with input settings
    :param gamma_max: (optional) maximum value specific weight soil
    :param pwp_level: (optional) pore water level in NAP
    """

    # read the cpt files
    import cpt_module
    import log_handler

    # Define log file
    log_file = log_handler.LogFile(output_folder, index)

    jsn = {"scenarios": []}
    i = 0
    for idx_cpt in range(len(cpt_BRO)):
        log_file.info_message("analysis started for: " + cpt_BRO[idx_cpt]["id"])
        cpt = cpt_module.CPT(output_folder, log_file)
        cpt.parse_bro(cpt_BRO[idx_cpt])
        cpt.qt_calc()
        cpt.gamma_calc(gamma_max, method=methods["gamma"])
        cpt.rho_calc()
        cpt.stress_calc(pwp_level)
        cpt.lithology_calc()
        cpt.IC_calc()
        cpt.vs_calc(method=methods["vs"])
        cpt.damp_calc(method=methods["OCR"])
        cpt.poisson_calc()
        cpt.merge_thickness(float(input_dictionary["MinLayerThickness"]))
        cpt.add_json(jsn, i)
        if make_plots:
            cpt.write_csv()
            cpt.plot_cpt()
            cpt.plot_lithology()
        i += 1
        log_file.info_message("analysis succeeded for: " + cpt_BRO[idx_cpt]["id"])
    cpt.update_dump_json(jsn, input_dictionary, index)
    log_file.close()
    return


def analysis(properties, methods_cpt, output, plots):
    import sys

    # number of points
    nb_points = len(props["Source_x"])

    # for each calculation point
    for i in range(nb_points):
        # read BRO data base
        inpt = {"BRO_data": properties["BRO_data"],
                "Source_x": properties["Source_x"][i], "Source_y": properties["Source_y"][i],
                "Radius": 1000}
        cpts = bro.read_bro(inpt)
        if not cpts:
            print("# WARNING #: No CPTS in this coordinate point: " + properties["Source_x"][i] + " " + properties["Source_y"][i])
            continue
        # process cpts
        read_cpt(cpts, methods_cpt, output, properties, plots, i)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json', help='input JSON file', required=True)
    parser.add_argument('-o', '--output', help='location of the output folder', required=True)
    parser.add_argument('-p', '--plots', help='make plots', required=False, default=False)
    parser.add_argument('-m', '--methods', help='define methods for CPT correlations', required=False, default=False)
    args = parser.parse_args()

    key = set_key()
    props = read_json(args.json)
    methods = define_methods(args.methods)

    # do analysis
    analysis(props, methods, args.output, args.plots)

