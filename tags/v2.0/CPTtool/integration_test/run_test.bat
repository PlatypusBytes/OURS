call cd ..\..\
call starting_point.cmd
call cd .\CPTtool\integration_test
call activate ours_environment
call python -m unittest discover .\