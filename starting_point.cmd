CALL conda env create -f env_cpttool.yml
CALL activate ours_env
CALL conda env update -f env_cpttool.yml
CALL activate ours_env
CALL pip install teamcity-messages==1.25