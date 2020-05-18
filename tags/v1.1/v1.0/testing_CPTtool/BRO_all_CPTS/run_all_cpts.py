import sys
sys.path.append(r"../../CPTtool")
sys.path.append(r"../../CPTtool/shapefiles")
import bro
import os
import pickle
import cpt_tool
import log_handler


def test_all_cpts(bro_path, output_folder):

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # define methods
    methods = {"gamma": "Robertson",
               "vs": "Robertson",
               "OCR": "Robertson"
               }

    # define properties
    properties = {"Name": "een naam",
                  "MaxCalcDist": "25.0",
                  "MaxCalcDepth": "30.0",
                  "MinLayerThickness": "0.5",
                  "SpectrumType": "1",
                  "LowFreq": "1",
                  "HighFreq": "63",
                  "CalcType": "1",
                  "Source_x": ["159191"],
                  "Source_y": ["435635"],
                  "Receiver_x": ["104883"],
                  "Receiver_y": ["478457"],
                  "BRO_data": "./bro_dataset/brocpt.xml"
                  }

    # index file
    with open(os.path.join(bro_path, "brocpt.idx"), "rb") as f:
        idx_file = pickle.load(f)

    # xml file
    xml_file = os.path.join(bro_path, "brocpt.xml")

    # read all cpts single
    for j in range(len(idx_file[1])):
        print(j)
        sys.stderr.write(str(j) + "\n")

        jsn = {"scenarios": []}
        i = j

        indices = [idx_file[1][i][2:4]]
        # call bro
        cpt = bro.read_bro_xml(xml_file, indices)

        # check if empty
        cpt = list(filter(None, cpt))
        if not cpt:
            continue

        # process the file if not empty
        log_file = log_handler.LogFile(output_folder, i)
        cpt_tool.read_cpt(cpt, methods, output_folder, properties, False, 0, log_file, jsn, 0)
        log_file.close()

        sys.stderr.flush()

    return


if __name__ == "__main__":
    sys.stderr = open('./log.txt', 'w')
    test_all_cpts("./bro_dataset", "./results")
    sys.stderr.close()
