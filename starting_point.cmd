CALL conda env create -f env_cpttool.yml
CALL activate ours
CALL conda env update -f env_cpttool.yml
CALL activate ours
CALL pip install pyqt5==5.12.1
CALL pip install teamcity-messages==1.24