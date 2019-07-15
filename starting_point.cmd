CALL conda env create -f env_cpttool.yml
CALL activate ours
CALL conda env update -f env_cpttool.yml
CALL activate ours
CALL pip install teamcity-messages==1.25