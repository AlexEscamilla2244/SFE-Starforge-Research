## Density datacubes

### Requirements

* numpy
* matplotlib
* pickle
* h5py

### Explanation

In this activity, we'll be using simulation data of dark matter. To start, we'll need to download the data: either do this on the command line (with wget https://github.com/usc-cosmolab/hackspace/blob/master/data/halos_short.bin?raw=true -O halos_short.bin) or follow the Github link and download the file from there. 

The simulation file is an N-body simulation containing ~8.6 billion particles (2048^3) and is meant to study how the dark matter particles clump together into halos.

### What next?

Run the provided Python script, which will open the simulation file and plot an example snapshot of the data. Once you've done that, consider making some adjustments before binning the points and saving the datacube for viewing in the VizLab: 

* Use the mass of the subhalos ('mvir' keyword) to get a subset of the simulation - those that are low mass, high mass, or something in between. Where are those halos located?
* Use the radius of the subhalos ('rvir' keyword) to get a subset of the simulation. Like before, are there any trends you see?
* (space willing) Adjust the dimensions of the binning process to change how much detail is preserved in the datacube. How much information is saved in a 128^3 cube? How about 64^3?

### Author's note

This exercise is based off of Python exercises from Ethan Nadler, which you can find here: https://github.com/eonadler/Colab-Notebooks/blob/main/Week_4_Exploring_N_body_Simulations.ipynb