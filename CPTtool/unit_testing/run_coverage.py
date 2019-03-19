import os
if __name__ == "__main__":
    os.system(r"coverage run -a test_cpt_tool.py")
    os.system(r"coverage run -a test_cpt_module.py")
    os.system(r"coverage run -a test_robertson.py")
    os.system(r"coverage report")
    os.system(r"coverage html")
