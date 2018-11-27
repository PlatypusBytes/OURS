..\..\Python_37\Python.exe -m coverage run -a test_cpt_tool.py
..\..\Python_37\Python.exe -m coverage run -a test_cpt_module.py
..\..\Python_37\Python.exe -m coverage report test_cpt_tool.py test_cpt_module.py
..\..\Python_37\Python.exe -m coverage html test_cpt_tool.py test_cpt_module.py
