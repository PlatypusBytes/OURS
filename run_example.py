import CPTtool

# input json file
file_input = "./test_folder/input.json"
# output folder location
output_folder = "./test_folder/output"
# creates plots: True / False
plots = True

# Runs CPTtool
props = CPTtool.cpt_tool.read_json(file_input)
methods = CPTtool.cpt_tool.define_methods(False)
settings = CPTtool.cpt_tool.define_settings(False)
CPTtool.cpt_tool.analysis(props, methods, settings, output_folder, plots)
