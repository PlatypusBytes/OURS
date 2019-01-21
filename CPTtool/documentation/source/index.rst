.. CPT tool documentation master file, created by
   sphinx-quickstart on Wed Aug 29 13:01:53 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CPT tool's documentation
========================

*Authors:* `Bruno Zuada Coelho <Bruno.ZuadaCoelho@deltares.nl>`_,  `Dirk de Lange <Dirk.deLange@deltares.nl>`_, `Eleni Smyrniou <Eleni.Smyrniou@deltares.nl>`_
           

*Version:* 1.0 beta

*License:* `Deltares <http://www.deltares.nl/>`_

*Last updated*: |today|


CPT tool
========

CPT tool determines soil profiles and soil properties from CPT files.

To run the code::   

	/> CPTool.exe -i <input_file> -o <output> -p <plots, optional>
	
The arguments have the following meaning:

	* *-i*: path to the input JSON file
	* *-o*: output path to the generation of results
	* *-p*: (optional) plots the results

Example: 

.. code-block:: python

    CPTTool.exe -i ./ground.json -o ./results -p True

    
The JSON file has the following structure:
   
.. code-block:: json

    {
       "Name": "name of the project",
       "MaxCalcDist": "25.0",
       "MaxCalcDepth": "30.0",
       "MinLayerThickness": "0.5",
       "SpectrumType": "1",
       "LowFreq": "1",
       "HighFreq": "63",
       "CalcType": "1",
       "Source_x": "111111.11",
       "Source_y": "222222.22",
       "Receiver_x": "111111.11",
       "Receiver_y": "222222.22",
       "BRO_data": "./cpts/"
    }

where the attributes have the following meaning:

       * *Name*: name of the project
       * *MaxCalcDist*: Maximum distance of calculation 
       * *MaxCalcDepth*: Maximum depth of calculation
       * *MinLayerThickness*:  Minimum layer thickness for the vertical discretisation
       * *SpectrumType*: Spectrum type 1=Octave bands 2=One-third octave bands
       * *LowFreq*: Minimum frequency of interest
       * *HighFreq*: Maximum frequency of interest
       * *CalcType*: Type of calculation 1=2D-FEM; 2=3D-FEM
       * *Source_x*: RD X coordinate of the source
       * *Source_y*: RD Y coordinate of the source 
       * *Receiver_x*: RD X coordinate of the receiver 
       * *Receiver_y*: RD Y coordinate of the receiver 
       * *BRO_data*: path to the BRO xml file


    
Content
=======

.. toctree::
   :maxdepth: 2
   
   cpt_tool
   cpt_module
   robertson
