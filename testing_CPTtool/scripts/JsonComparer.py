from CPTtool import cpt_tool
import json


def read_json(path):
    with open(path+'results.json') as f:
        data = json.load(f)
    return data


def json_comparer(d1, d2, path=""):
    for k in d1.keys(): # are there senerios missing?
        if k not in d2:
            print(path, ":")
            print(k + " as key not in d2", "\n")
        else:
            if type(d1[k]) is dict: # is the next a dictionary?
                if path == "":
                    path = k
                else:
                    path = path + "->" + k
                json_comparer(d1[k], d2[k], path)
            else:
                if d1[k] != d2[k]:
                    print('FAILS IN PATH',path, "WITH VALUES :")
                    print(" - ", k, " : ", d1[k])
                    print(" + ", k, " : ", d2[k])
                else:
                    print('SUCCEED IN PATH', path, "WITH VALUES :")
                    print(" - ", k, " : ", d1[k])
                    print(" + ", k, " : ", d2[k])
    return


if __name__ == '__main__':
    cpt_results = cpt_tool.read_cpt("..\cpts_tests", cpt_tool.set_key(), '..\case_results', 0.5,False)
    reference_json = read_json('../results_reference/')
    case_json = read_json('../case_results/')
    General_Equality = (reference_json == case_json)
    print(General_Equality)
    json_comparer(reference_json,case_json)
