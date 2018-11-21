.. CPT tool documentation master file, created by
   sphinx-quickstart on Wed Aug 29 13:01:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CPT tool's documentation
========================

*Author:* `Bruno Zuada Coelho <Bruno.ZuadaCoelho@deltares.nl>`_

*Version:* 0.2 beta

*License:* `Deltares <http://www.deltares.nl/>`_

*Last updated*: |today|


CPT tool
========

CPT tool determines soil profiles and soil properties from CPT files.

To run the code::   

	/> CPTool.exe -k <key> -c <cpt> -o <output> -t <thickess> 
	
The arguments have the following meaning:

	* *-k*:  key of cpt file
	* *-c*: path to the cpt folder
	* *-o*: output path to the generation of results
	* *-t*: minimum layer thickness
	
        
Content
=======

.. toctree::
   :maxdepth: 2
   
   cpt_tool
   cpt_module
   robertson
