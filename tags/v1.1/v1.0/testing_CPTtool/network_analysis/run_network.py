import sys
sys.path.append(r"../../CPTtool")
import os
import cpt_tool


def test_network(bro_path, output_folder, summary_file):
    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # summary file
    fo = open(summary_file, "w")
    fo.write("Analysis started\n")

    # read coordinates
    with open("./coords_network.csv", "r") as f:
        coords = f.read().splitlines()
    coords = [i.split(";") for i in coords[1:]]

    # define methods
    methods = {"gamma": "Robertson",
               "vs": "Robertson",
               "OCR": "Mayne",
               "radius": 600,
               }

    # run analysis for each coordinate point
    for i, c in enumerate(coords):

        sys.stderr.write(str(c) + "\n")

        # define properties
        properties = {"Name": "asd",
                      "MaxCalcDist": "25.0",
                      "MaxCalcDepth": "30.0",
                      "MinLayerThickness": "0.5",
                      "SpectrumType": "1",
                      "LowFreq": "1",
                      "HighFreq": "63",
                      "CalcType": "1",
                      "Source_x": [c[0]],
                      "Source_y": [c[1]],
                      "Receiver_x": [c[0]],
                      "Receiver_y": [c[1]],
                      "BRO_data": bro_path,
                      }

        # run cpt tool
        fo.write("Analysis started for index: " + str(i) + " coordinate: " + str(c[0]) + " " + str(c[1]) + "\n")
        fo.flush()
        cpt_tool.analysis(properties, methods, os.path.join(output_folder, str(i) + "_" + str(c[0]) + "_" + str(c[1])), False)
        fo.write("Analysis done for index: " + str(i) + " coordinate: " + str(c[0]) + " " + str(c[1]) + "\n")
        fo.flush()

    fo.write("Analysis finished\n")
    return


if __name__ == "__main__":
    sys.stderr = open('log.txt', 'w')
    test_network("../../bro_dataset/brocpt.xml", "./results", "summary.txt")
    sys.stderr.close()
