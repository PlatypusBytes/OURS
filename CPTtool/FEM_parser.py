def cpt_to_fem(input_json, cpt_input, output_folder):
    """
    transform the json output of the CPTtool into input for the FEM analysis
    Writes a JSON file for each scenario

    :param input_json: JSON file with the input definitions
    :param cpt_input: JSON file with the input values (result from the CPTtool)
    :param output_folder: folder where the input files for the FEM will be saved
    """
    import os
    import json

    # if output does not exist: create
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # read input
    with open(input_json, "r") as f:
        input_data = json.load(f)

    # read input CPTtool
    with open(cpt_input, "r") as f:
        cpt_data = json.load(f)

    # write a file for each scenario
    for sce in range(len(cpt_data["scenarios"])):
        output = {"MaxCalcDist": float(input_data["MaxCalcDist"]),
                  "MaxCalcDepth": float(input_data["MaxCalcDist"]),
                  "MinLayerThickness": float(input_data["MinLayerThickness"]),
                  "LowFreq": float(input_data["LowFreq"]),
                  "HighFreq": float(input_data["HighFreq"]),
                  "Ground": {"Lithology": cpt_data['scenarios'][sce]["data"]["lithology"],
                             "Depth": [-i for i in cpt_data['scenarios'][sce]["data"]["depth"]],
                             "E": cpt_data['scenarios'][sce]["data"]["E"],
                             "v": cpt_data['scenarios'][sce]["data"]["v"],
                             "rho": cpt_data['scenarios'][sce]["data"]["rho"],
                             "damping": cpt_data['scenarios'][sce]["data"]["damping"],
                             "var_E": cpt_data['scenarios'][sce]["data"]["var_E"],
                             "var_v": cpt_data['scenarios'][sce]["data"]["var_v"],
                             "var_rho": cpt_data['scenarios'][sce]["data"]["var_rho"],
                             "var_damping": cpt_data['scenarios'][sce]["data"]["var_damping"],
                             }}
        # dump json file
        with open(os.path.join(output_folder, cpt_data['scenarios'][sce]["Name"].replace(" ", "_") + ".json"), "w") as fo:
            json.dump(output, fo, indent=4)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json', help='input JSON file', required=True)
    parser.add_argument('-c', '--cpt', help='CPTtool JSON file', required=True)
    parser.add_argument('-o', '--output', help='location of the output folder', required=True)
    args = parser.parse_args()

    cpt_to_fem(args.json, args.cpt, args.output)
