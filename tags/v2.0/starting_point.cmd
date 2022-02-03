CALL conda env create -f env_cpttool.yml
CALL activate ours_environment
CALL conda env update -f env_cpttool.yml
CALL activate ours_environment
CALL pip install teamcity-messages==1.30