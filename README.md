# Adaptive human behavior and delays in information availability autonomously modulate epidemic waves

This repository contains all code to run the behavioral-epidemiological infectious disease model, developed in the manuscript ["Adaptive human behavior and delays in information availability autonomously modulate epidemic waves"](https://www.medrxiv.org/cgi/content/short/2024.11.23.24317838v1), published in PNAS Nexus, 2025. The Python code was developed using Python 3.11.

The key file is model.py, which employs the just-in-time compiler [Numba](https://numba.pydata.org) to speed up the computation of the dynamics of the behavioral SIR model. To run the behavioral SEIR model, use the file model_extended.py.

All figures and analyses in the manuscript can be recreated by running the files 
- plot_basic_figures.py,
- sensitivity_analysis_2d.py,
- sensitivity_analysis_4d.py,
  
for the behavioral SIR model, as well as
- plot_basic_figures_extended.py,
- sensitivity_analysis_2d_extended.py,
- sensitivity_analysis_4d_extended.py
  
for the behavioral SEIR model.

