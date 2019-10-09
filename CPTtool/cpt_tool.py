import argparse
import log_handler
import bro
import sys
import tools_utils


def define_methods(input_file):
    """
    Defines correlations for the CPT

    :param input_file: json file with methods
    :return: methods: dictionary with the methods for CPT correlations
    """
    import os
    import json

    # possible keys:
    keys = ["gamma", "vs", "OCR", "radius"]
    # possible key-values
    gamma_keys = ["Robertson", "Lengkeek", "all"]
    vs_keys = ["Robertson", "Mayne", "Andrus", "Zang", "Ahmed", "all"]
    OCR_keys = ["Mayne", "Robertson"]
    rad = 500.

    # if no file is available: -> uses default values
    if not input_file:
        methods = {"gamma": gamma_keys[0],
                   "vs": vs_keys[0],
                   "OCR": OCR_keys[0],
                   "radius": rad,
                   }
        return methods

    # check if input file exists
    if not os.path.isfile(input_file):
        print("File with methods definition does not exist")
        sys.exit(4)

    # if the file is available
    with open(input_file, "r") as f:
        data = json.load(f)

    # # checks if the keys are correct
    for i in data.keys():
        if not any(i in k for k in keys):
            print("Error: Key " + i + " is not known. Keys must be: " + ', '.join(keys))
            sys.exit(5)

    # check if the key-values are correct for gamma
    if not any(data["gamma"] in k for k in gamma_keys):
        print("Error: gamma key is not known. gamma keys must be: " + ', '.join(gamma_keys))
        sys.exit(5)
    # check if the key-values are correct for vs
    if not any(data["vs"] in k for k in vs_keys):
        print("Error: gamma key is not known. vs keys must be: " + ', '.join(vs_keys))
        sys.exit(5)
    # check if the key-values are correct for OCR
    if not any(data["OCR"] in k for k in OCR_keys):
        print("Error: gamma key is not known. OCR keys must be: " + ', '.join(OCR_keys))
        sys.exit(5)
    # check if radius is a float

    if not isinstance(data["radius"], (int, float)):
        print("Error: radius is not known. must be a float")
        sys.exit(5)

    methods = {"gamma": data["gamma"],
               "vs": data["vs"],
               "OCR": data["OCR"],
               "radius": float(data["radius"])
               }
    return methods


def read_json(input_file):
    """
    Reads input json file

    :param input_file: json file with the input values
    :return: data: dictionary with the input files
    """
    import os
    import json

    # check if file exits
    if not os.path.isfile(input_file):
        print("Input JSON file does not exist")
        sys.exit(-3)

    # read file
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def read_cpt(cpt_BRO, methods, output_folder, input_dictionary, make_plots, index_coordinate, log_file, jsn,
             scenario, gamma_max=22, pwp_level=0):
    """
    Read CPT

    Read and process cpt files: GEF format

    Parameters
    ----------
    :param cpt_BRO: cpt information from the BRO
    :param methods: Methods for the CPT correlations
    :param output_folder: Folder to save the files
    :param input_dictionary: Dictionary with input settings
    :param make_plots: Bool to make plots
    :param index_coordinate: Index of the calculation coordinate point
    :param log_file: Log file for the analysis
    :param jsn: dictionary with the scenarios
    :param scenario: scenario number
    :param gamma_max: (optional) maximum value specific weight soil
    :param pwp_level: (optional) pore water level in NAP
    :return: return_cpt: list with the processed cpt objects
    """

    # read the cpt files
    import cpt_module

    # dictionary for the results
    results_cpt = {}
    for idx_cpt in range(len(cpt_BRO)):
        # add message to log file
        log_file.info_message("Reading CPT: " + cpt_BRO[idx_cpt]["id"])

        # initialise CPT module
        cpt = cpt_module.CPT(output_folder)
        # read data from BRO
        data_quality = cpt.parse_bro(cpt_BRO[idx_cpt])
        # check data quality from the BRO file
        if data_quality is not True:
            # If the quality is not good skip this cpt file
            log_file.error_message(data_quality)
            continue
        # smooth data
        cpt.smooth()
        # compute qc
        cpt.qt_calc()
        # compute unit weight
        cpt.gamma_calc(gamma_max, method=methods["gamma"])
        # compute density
        cpt.rho_calc()
        # compute stresses: total, effective and porewater pressures
        cpt.stress_calc(pwp_level)
        # compute lithology
        cpt.lithology_calc()
        # compute IC
        cpt.IC_calc()
        # compute shear wave velocity and shear modulus
        cpt.vs_calc(method=methods["vs"])
        # compute damping
        cpt.damp_calc(method=methods["OCR"])
        # compute Poisson ratio
        cpt.poisson_calc()
        # make the plots (optional)
        if make_plots:
            cpt.write_csv()
            cpt.plot_cpt()
            cpt.plot_lithology()
        # update scenario
        results_cpt.update({cpt_BRO[idx_cpt]["id"]: cpt})
        # add to log file that the analysis is successful
        log_file.info_message("Analysis succeeded for: " + cpt_BRO[idx_cpt]["id"])

    # perform interpolation
    result_interp = tools_utils.interpolation(results_cpt, [input_dictionary['Receiver_x'][index_coordinate],
                                                            input_dictionary['Receiver_y'][index_coordinate]])

    # merge the layers thickness
    depth_json, indx_json, lithology_json = tools_utils.merge_thickness(result_interp, float(input_dictionary["MinLayerThickness"]))
    # add results to the dictionary
    jsn = tools_utils.add_json(jsn, scenario, depth_json, indx_json, lithology_json, result_interp)
    return jsn


def analysis(properties, methods_cpt, output, plots):
    """
    Analysis of CPT

    Extracts the CPT from the BRO PDOK database, based on coordinate location and processes the cpt

    :param properties: JSON file with the properties of the analysis: location
    :param methods_cpt: methods to use for the CPT interpretation
    :param output: path for the output results
    :param plots: boolean create the plots
    :return:
    """
    # number of points
    nb_points = len(properties["Source_x"])
    # probability of scenarios
    prob = []

    # for each calculation point
    for idx in range(nb_points):

        # variables
        jsn = {"scenarios": []}  # json dictionary
        scenario = 0  # scenario number

        # Define log file
        log_file = log_handler.LogFile(output, idx)
        log_file.info_message("Analysis started for coordinate point: (" + properties["Source_x"][idx] + ", "
                              + properties["Source_y"][idx] + ")")

        # read BRO data base
        inpt = {"BRO_data": properties["BRO_data"],
                "Source_x": float(properties["Source_x"][idx]), "Source_y": float(properties["Source_y"][idx]),
                "Radius": float(methods_cpt["radius"])}
        cpts = bro.read_bro(inpt)

        results = {}
        # check points within polygons
        cpts_polygons = {}
        polygons_names = []
        for zone in cpts['polygons']:
            cpts_polygons.update({zone: {"data": list(filter(None, cpts['polygons'][zone]['data'])),
                                         "perc": cpts['polygons'][zone]['perc']
                                         }
                                  })

            for c in cpts_polygons[zone]["data"]:
                polygons_names.append(c["id"])

        # process cpts polygons
        results.update({"polygons": {}})
        for zone in cpts_polygons:
            # remove the nones
            data = list(filter(None, cpts['polygons'][zone]['data']))
            if data:
                jsn = read_cpt(data, methods_cpt, output, properties, plots, idx, log_file, jsn, scenario)
                results["polygons"].update({zone: True})
                prob.append(cpts['polygons'][zone]['perc'])
                jsn["scenarios"][scenario].update({"coordinates": [properties["Receiver_x"][idx], properties["Receiver_y"][idx]],
                                                   "probability": round(prob[-1], 2)})
                scenario += 1

        # check points within circle
        cpts_circle = list(filter(None, cpts['circle']['data']))
        # get cpts that are not in polygons
        circle_names = []
        circle_idx = []
        for j, c in enumerate(cpts_circle):
            circle_names.append(c["id"])
            circle_idx.append(j)

        circle_names = [c["id"] for c in cpts_circle]
        # process only the ones that are not part of polygons
        names_diff = list(set(circle_names) - set(polygons_names))
        # get the indexes of the circle cpts to be processed
        circle_idx = [circle_names.index(n) for n in names_diff]
        cpts_circle = [cpts_circle[j] for j in circle_idx]

        # remove the nones
        data = list(filter(None, cpts_circle))

        # get indexes of
        results.update({"circle": []})
        if data:
            # if data exists in the circle
            jsn = read_cpt(data, methods_cpt, output, properties, plots, idx, log_file, jsn, scenario)
            results["circle"] = True
            jsn["scenarios"][scenario].update({"coordinates": [properties["Receiver_x"][idx], properties["Receiver_y"][idx]],
                                               "probability": round(1. - sum(prob), 2)})
            scenario += 1
        elif jsn["scenarios"]:
            # if circle is empty and polygons exist: update probability of polygons
            for i in range(len(jsn["scenarios"])):
                jsn["scenarios"][i]["probability"] = jsn["scenarios"][i]["probability"] / sum(prob)

        # check if cpts have data or are all empty: this mean that this point has no data
        if not results["circle"] and not results["polygons"]:
            log_file.error_message("No data in this coordinate point")
            log_file.info_message("Analysis finished for coordinate point: (" + str(properties["Source_x"][idx]) + ", "
                                  + str(properties["Source_y"][idx]) + ")")
            log_file.close()
            continue

        # dump json
        tools_utils.dump_json(jsn, idx, output)

        # processed cpts
        log_file.info_message("Analysis finished for coordinate point: (" + str(properties["Source_x"][idx]) + ", "
                              + str(properties["Source_y"][idx]) + ")")
        log_file.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json', help='input JSON file', required=True)
    parser.add_argument('-o', '--output', help='location of the output folder', required=True)
    parser.add_argument('-p', '--plots', help='make plots', required=False, default=False)
    parser.add_argument('-m', '--methods', help='define methods for CPT correlations', required=False, default=False)
    args = parser.parse_args()

    # reads input json file
    props = read_json(args.json)
    # define methods for the analysis of CPT
    methods = define_methods(args.methods)

    # do analysis
    analysis(props, methods, args.output, args.plots)
