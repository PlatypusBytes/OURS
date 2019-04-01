conda env create -f env_cpttool.yml
CALL activate ours
conda env update -f env_cpttool.yml
pip install pyqt5==5.12.1
pip install teamcity-messages==1.24