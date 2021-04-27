====
gbpy
====

.. image:: https://img.shields.io/travis/spatala/gbpy.svg
        :target: https://travis-ci.org/spatala/gbpy

.. image:: https://img.shields.io/pypi/v/gbpy.svg
        :target: https://pypi.python.org/pypi/gbpy


Python package for doing science.

* Free software: 3-clause BSD license
* Documentation:  https://spatala.github.io/gbpy.
* GitHub: https://github.com/spatala/gbpy
* PyPI: https://pypi.org/project/GBpy/
* Tutorials: https://spatala.github.io/gbpy/Tutorials/index.html


How to Use This Package:
========================
1.  **To install the stable version of gbpy:**      
    
    .. code-block:: console
                
        $ pip install gbpy
                
2. **Import the package**
	.. code-block:: pycon
		>>> import gbpy
3. ** call the function by using**
	.. code-block:: pycon
		>>> gbpy.<name_of_the_function>
	* For example to find the minimum energy structure of GB $\Sigma (0,0,1)(0,0,\bar{1})$ in Mg
		.. code-block:: pycon
		>>> element = 'Mg_sun'
		>>> CohEng = -1.5287  # calculated from in.cohesive
		>>> Tm = 923.1
		>>> lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
		>>> gb_name = 'gb_attr_Mg_sun_S7_1_N1_0_0_1_N2_0_0_-1'
		>>> gbpy.MC_MD(gb_name, element, CohEng, Tm, lammps_exe_path)

Consult the `documentation <https://spatala.github.io/gbpy/>`__ for further details.
        
        
Prerequisites:
==============
                
1. install ``numpy`` from `here. <http://www.numpy.org/>`__
                
2. install ``scipy`` from `here. <http://www.scipy.org/>`__

3. install ``pyhull`` from `here. <https://pythonhosted.org/pyhull/>`__

4. install ``ovito`` from `here. <https://www.ovito.org/>`__

5. install ``sympy`` from `here. <https://www.sympy.org/>`__

                
Cite GBpy:
========================


                
Credits:
========
gbpy is written by:
                
* `Srikanth Patala <spatala@ncsu.edu>`__
* `Leila Khalili <lkhalil@ncsu.edu>`__
* `Patala Research Group <http://research.mse.ncsu.edu/patala/>`__.
        
Copyright (c) 2015,  Leila Khalili and Srikanth Patala.