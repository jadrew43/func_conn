# func_conn 

Output of functional connectivity measures located in `/analysis_output`.

Use `exp1Stats.py`/`exp2Stats.py` to produce time-varying autoregressive (AR) plots, graphical representation of statistics, and histogram containing averaged AR coefficients across bootstraps and across regions of interest, for Experiment 1 (Original Study) / Experiment 2 (Validation Study). Significant connections require 95/100 (`thresh_bs=95`) bootstraps to meet or exceed the 90th percentile value (`thresh_90th=0.11` for experiments 1&2) of the experiment's averaged AR coefficients.

The class for defining the different experiments, as well as various functions for creating the statistical graphs, are located in `graphStats_ROIs.py`.

Comparable Experimental Conditions for Experiment 1 / Experiment 2:  
 - Maintain Space - LL / MS  
 - Switch Space - LR / SS  
 - Maintain Pitch - UU / MP  
 - Switch Pitch - UD / SP  

Uncommon Experimental Conditions for Experiment 1 
 - Space -> Pitch - LX 
 - Pitch -> Space - UX 

Uncommon Experimental Conditions for Experiment 2
 - Maintain Both - MB
 - Switch Both - SB

Tools and functions for plotting the time-varying AR plots located in `plotting.py` which uses `labels.pkl`, containing names of different cortical ROI labels according to FreeSurfer. 
