call conda activate OURS
python -m coverage run -a test_cpt_tool.py
python -m coverage run -a test_cpt_module.py
python -m coverage run -a test_robertson.py
python -m coverage report
python -m coverage html