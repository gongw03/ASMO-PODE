# ASMO-PODE
This is a standalone version of ASMO-PODE, a surrogate based optimization and distribution estimation algorithm.

Quick start: please run banana2D/banana2D\_ASMOPODE.py to start your first run. For more information about ASMO-PODE, please read the paper. And please cite it if you use the code in your own research.
Gong, W., & Duan, Q. (2017). An adaptive surrogate modeling-based sampling strategy for parameter optimization and distribution estimation (ASMO-PODE). Environmental Modelling & Software, 95, 61–75. https://doi.org/10.1016/j.envsoft.2017.05.005

Many test cases with the test function banana2D.
1. banana2D\_Metropolis.py: Metropolis algorithm, a classical Markov Chain Monte Carlo algorithm.
2. banana2D\_AM.py: Adaptive Metropolis algorithm.
3. banana2D\_DRAM.py: Delayed Rejection Adaptive Metropolis algorithm.
4. banana2D\_ASMOPODE.py: This algorithm ASMO-PODE.

ASMOPODE.py support parallel evaluation of multiple Markov Chains on surrogate model. Just set Parallel = True.
However, because module object cannot be pickled, although Metropolis.py AM.py and DRAM.py support parallel compution, the banana2D.py module cannot be used parallely with them.

Note that must compile the c-based Gaussian Processes Regression libs.

Step 1: cd src

Step 2: install swig in your python. For an example, conda install swig (if you use anaconda). 
Better to install numba to accelerate Metropolis. For an example, conda install numba (if you use anaconda)

Step 3: python download\_numpy.i.py (download corresponding numpy.i file for swig)

Step 4: python setup\_cgp.py build 

Step 5: cp build/lib.your.system.xxx/\_cgp.cpython.xxx.so . (file name may be different for different systems)

Done! Let's try ASMO-PODE now!

Step 6: cd ../banana2D

python banana2D\_ASMOPODE.py

Other files in directory banana2D:
1. banana2D.py: the 2D distribution function banana2D.
2. banana2D.txt: the parameter name, lower bound, upper bound.
