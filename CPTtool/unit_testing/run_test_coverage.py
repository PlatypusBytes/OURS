import os
import sys
os.system(os.path.join(sys.exec_prefix, r"Scripts/coverage") + " run -a test_cpt_tool.py")
os.system(os.path.join(sys.exec_prefix, r"Scripts/coverage") + " run -a test_cpt_module.py")
os.system(os.path.join(sys.exec_prefix, r"Scripts/coverage") + " run -a test_robertson.py")
os.system(os.path.join(sys.exec_prefix, r"Scripts/coverage") + " run -a test_tools_utils.py")
os.system(os.path.join(sys.exec_prefix, r"Scripts/coverage") + " report")
os.system(os.path.join(sys.exec_prefix, r"Scripts/coverage") + " html")
