from CPTtool import cpt_tool

# input json file
file_input = "./example/input.json"
# output folder location
output_folder = "./example/output"
# creates plots: True / False
plots = False

# Runs CPTtool
props = cpt_tool.read_json(file_input)
methods = cpt_tool.define_methods(False)
settings = cpt_tool.define_settings(False)
cpt_tool.analysis(props, methods, settings, output_folder, plots)
