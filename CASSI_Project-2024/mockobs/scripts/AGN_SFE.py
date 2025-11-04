#!/usr/bin/env python
"""
Generates a range of values of star formation efficiency 

Usage: e_ff.py <file1> <file2> ... [options]

Options:
  -h --help                  Show this screen
  --BoxSize=<BX>             Box size of the simulation (defaults to 100)
  --Starting_Range=<R_i>     Starting range of density cutoff (defaults to 65)
  --Ending_Range=<R_f>       Ending range of density cutoff (defaults to 2500)
  --Resolution=<R>           Resolution of simulation (defaults to 1024)
  --output_path=<PATH>       Output path for data (defaults to /E_FF_range in file1 directory)
  --AGN=<AGN>                Set number of AGN contaminants per degree squared (default is 9)
  --Distance=<D>
"""

from scipy.spatial import KDTree
import pandas as pd
from astropy.constants import G
import astropy.units as u
import os
import h5py
import numpy as np
from docopt import docopt
import pathlib

# Set a fixed random seed for reproducibility
np.random.seed(42)


def read_yso_data(file_path):
    """Read YSO coordinates from the given file.
    
    Opens an HDF5 file and reads the 'X_pc' and 'Y_pc' datasets which contain
    the X and Y coordinates (in parsecs) of the YSOs.
    
    Returns:
        tuple: (YSO_X, YSO_Y) arrays of coordinates.
    """
    with h5py.File(file_path, "r") as f:
        # Read X and Y coordinates from the file
        YSO_X = f["X_pc"][:]
        YSO_Y = f["Y_pc"][:]
    return YSO_X, YSO_Y


def read_dust_data(file_path):
    """Read dust data and create flattened density and dust coordinate arrays.
    
    Opens the HDF5 file and reads:
      - X and Y coordinate grids,
      - Surface density data (in Msun/pc^2).
    
    It then flattens the surface density and coordinate arrays for further analysis.
    
    Returns:
        tuple: (X, Y, Surface_Den, new_den, dust_coords) where:
            - X, Y: Original coordinate grids.
            - Surface_Den: 2D array of surface density.
            - new_den: Flattened surface density with proper units.
            - dust_coords: 2D array of flattened X, Y coordinates.
    """
    with h5py.File(file_path, "r") as f:
        X = f["X_pc"][:]
        Y = f["Y_pc"][:]
        Surface_Den = f["SurfaceDensity_Msun_pc2"][:]
    # Flatten the density array and assign units of solar mass per pc^2
    new_den = Surface_Den.flatten() * (u.solMass / u.pc**2)
    # Combine the flattened X and Y arrays into a 2-column coordinate array
    dust_coords = np.c_[X.flatten(), Y.flatten()]
    return X, Y, Surface_Den, new_den, dust_coords


def inject_agn(Surface_Den, X, Y, Distance):
    """Inject AGN contamination based on the densest region in the dust data.
    
    This function calculates the expected number of AGN contaminants using a Poisson 
    process and then injects AGN by sampling coordinates around the region of highest 
    surface density. Note: 'Distance' is assumed to be a global variable.
    
    Args:
        Surface_Den (array): 2D array of dust surface density.
        X, Y (arrays): 2D coordinate arrays corresponding to Surface_Den.
    
    Returns:
        tuple: (AGN_x, AGN_y) arrays containing the AGN coordinates.
    """
    # Calculate the squared small angle (10 pc in kpc) relative to the simulation distance
    small_angle = ((10 * u.pc).to(u.kpc) / (Distance * u.kpc))**2
    # AGN contaminants per degree squared (from literature/default)
    AGN_per_degree_squared = 9.0
    # Convert small angle from steradians to square degrees
    area_deg2 = small_angle * (180 / np.pi)**2
    # Expected number of AGN from a Poisson distribution
    lambda_agn = AGN_per_degree_squared * area_deg2
    N_AGN = np.random.poisson(lam=lambda_agn)

    # Identify the index of the maximum surface density (densest region)
    max_dens_idx = np.unravel_index(np.argmax(Surface_Den), Surface_Den.shape)
    dense_x, dense_y = X[max_dens_idx], Y[max_dens_idx]

    # Sample AGN positions normally distributed around the densest region (scale=5)
    AGN_x = np.random.normal(loc=dense_x, scale=5, size=N_AGN)
    AGN_y = np.random.normal(loc=dense_y, scale=5, size=N_AGN)
    return AGN_x, AGN_y


def combine_coordinates(YSO_X, YSO_Y, AGN_x, AGN_y):
    """Combine YSO and AGN coordinates into a single array.
    
    Concatenates the X and Y coordinate arrays of the YSOs and AGN to create a unified
    coordinate list for further processing.
    
    Args:
        YSO_X, YSO_Y (arrays): Arrays of YSO coordinates.
        AGN_x, AGN_y (arrays): Arrays of AGN coordinates.
    
    Returns:
        array: Combined 2D array of coordinates.
    """
    # Concatenate X coordinates of YSOs and AGN
    combined_x = np.concatenate([YSO_X, AGN_x])
    # Concatenate Y coordinates of YSOs and AGN
    combined_y = np.concatenate([YSO_Y, AGN_y])
    # Stack the combined X and Y arrays as columns
    return np.c_[combined_x, combined_y]


def compute_sfe_values(yso_surface_den, new_den, Box_Size, R_I, R_F, RES):
    """
    Loop over density cutoffs and compute star formation efficiency (SFE) values.
    
    For each density threshold from R_I to R_F, the function:
      - Counts the number of YSOs above the threshold.
      - Estimates the star formation rate.
      - Computes the gas mass and area above the threshold.
      - Calculates the free-fall time and the star formation efficiency.
    
    Returns:
        tuple: Lists containing SFE values, number of YSOs, gas mass, area, and free-fall times.
    """
    # Convert gravitational constant G to units: pc^3 / (Msun * Myr^2)
    new_G = G.to(u.pc**3 / (u.solMass * u.Myr**2)).value
    # Initialize lists for the computed parameters
    e_ff_range = []
    N_YSOS = []
    Mass_Gass = []
    Area_list = []
    tff = []
    Sfr_list = []


    
    # Define a unit surface density (1 Msun/pc^2)
    surface_unit = 1 * (u.solMass / u.pc**2)
    # Determine the effective grid cell length (Box_Size divided by 5)
    L = Box_Size / 5

    # Loop over a range of density cutoff values (with step=1)
    for i in np.arange(R_I, R_F, 1):
        # Set the threshold density for the current iteration
        threshold = i * surface_unit
        # Count the number of YSOs with surface density above the threshold
        num_yso_at_surface_den = (yso_surface_den > threshold).sum()
        # Estimate the star formation rate based on the number of YSOs (factor cancels here)
        StarRate = (((num_yso_at_surface_den * 0.5) / 0.5) * (u.solMass / u.Myr))

        # Create a mask selecting cells where the density exceeds the threshold
        Mask = new_den > (i * surface_unit)
        # Calculate the area per grid cell based on the simulation's resolution
        A_i = (L / RES)**2 * u.pc**2
        # Total area above threshold: sum of masked cells multiplied by area per cell
        A = (np.sum(Mask) * A_i).value
        # Total gas mass above threshold: sum of density values for masked cells multiplied by area per cell
        M_Gas = np.sum(A_i * new_den[Mask]).value

        # Check for non-physical conditions (avoid division by zero)
        if M_Gas <= 0 or A <= 0:
            T_ff = np.nan
            star_efficiency = np.nan
        else:
            # Calculate free-fall time (T_ff) from gravitational collapse considerations
            t_ff_num = (A ** (3 / 2)) * np.sqrt(np.pi)
            t_ff_denom = 8 * new_G * M_Gas
            T_ff = np.sqrt(t_ff_num / t_ff_denom)
            # Compute star formation efficiency: ratio of StarRate to (gas mass per free-fall time)
            star_efficiency = (StarRate / (M_Gas / T_ff)).value

        # Append the computed values for this density cutoff to the lists
        e_ff_range.append(star_efficiency)
        N_YSOS.append(num_yso_at_surface_den)
        Mass_Gass.append(M_Gas)
        Area_list.append(A)
        tff.append(T_ff)
        Sfr_list.append(StarRate)

    return e_ff_range, N_YSOS, Mass_Gass, Area_list, tff


def write_results(file1, OUTPATH, e_ff_range, N_YSOS, Mass_Gass, Area_list, tff):
    """
    Write the computed SFE values and associated data to an HDF5 file.
    
    Constructs an output filename based on the input YSO file name and writes the datasets 
    (SFE values, number of YSOs, gas mass, area, and free-fall times) to the file.
    
    Args:
        file1 (str): Path to the YSO file.
        OUTPATH (str): Output directory path.
        e_ff_range, N_YSOS, Mass_Gass, Area_list, tff (lists): Computed data arrays.
    """
    # Construct the output filename by replacing part of the input file name
    fname = os.path.basename(file1).replace(".YSOobjects.hdf5", ".AGN_e_ff_range.hdf5")
    # Determine the output directory: use provided OUTPATH or default to an "E_FF" subdirectory
    if OUTPATH:
        outdir = OUTPATH
    else:
        outdir = os.path.join(str(pathlib.Path(file1).parent.resolve()), "E_FF")
    # Create the output directory if it doesn't already exist
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Combine the directory and filename to get the full output file path
    imgpath = os.path.join(outdir, fname)

    # Write each computed dataset into the HDF5 file
    with h5py.File(imgpath, "w") as f:
        f.create_dataset("SFE_Values", data=e_ff_range)
        f.create_dataset("NYSOs", data=N_YSOS)
        f.create_dataset("M_Gas", data=Mass_Gass)
        f.create_dataset("Area", data=Area_list)
        f.create_dataset("T_ff", data=tff)
        f.create_dataset("SFR", data=Sfr_list)
    print("Output written to:", imgpath)


def SFE_values(path1, path2, Box_Size, R_I, R_F, OUTPATH, RES, Distance, return_results=False):
    """
    Pipeline to compute star formation efficiency (SFE) values.
    
    1. Reads YSO and dust data from the provided files.
    2. Injects AGN contamination into the data.
    3. Combines YSO and AGN coordinates.
    4. Uses a KDTree to assign dust surface density values to the combined coordinates.
    5. Computes the SFE over a range of density cutoffs.
    6. Either writes the results to an output HDF5 file or returns the results.
    
    Args:
        path1 (str): Path to the YSO file.
        path2 (str): Path to the dust file.
        Box_Size (int): Size of the simulation box.
        R_I (int): Starting density cutoff.
        R_F (int): Ending density cutoff.
        OUTPATH (str): Output directory path.
        RES (int): Simulation resolution.
        Distance (float): Distance parameter to vary the AGN injection.
        return_results (bool): If True, return computed data instead of writing to file.
    
    Returns:
        tuple: (e_ff_range, N_YSOS, Mass_Gass, Area_list, tff) if return_results is True.
    """
    # Read YSO coordinate data from file1
    YSO_X, YSO_Y = read_yso_data(path1)
    # Read dust data from file2, which includes surface density and coordinate grids
    X, Y, Surface_Den, new_den, dust_coords = read_dust_data(path2)

    # Inject AGN contaminants into the data based on dust density using the provided Distance
    AGN_x, AGN_y = inject_agn(Surface_Den, X, Y, Distance)
    # Combine YSO and AGN coordinates into a single array for analysis
    contaminated = combine_coordinates(YSO_X, YSO_Y, AGN_x, AGN_y)

    # Build a KDTree for efficient spatial matching using the dust coordinates
    T = KDTree(dust_coords)
    # Query the nearest dust coordinate for each contaminated position
    _, idx = T.query(contaminated)
    # Create a DataFrame mapping each contaminated coordinate to its dust surface density
    df = pd.DataFrame({
        "YSOs": contaminated[:, 0],
        "YSO Surface Den": new_den[idx]
    })
    # Extract the dust surface density corresponding to each YSO/AGN coordinate
    yso_surface_den = df["YSO Surface Den"]

    # Compute the star formation efficiency values over the density cutoff range
    e_ff_range, N_YSOS, Mass_Gass, Area_list, tff = compute_sfe_values(
        yso_surface_den, new_den, Box_Size, R_I, R_F, RES
    )

    # Depending on the flag, either return results or write to file.
    if return_results:
        return e_ff_range, N_YSOS, Mass_Gass, Area_list, tff
    else:
        write_results(path1, OUTPATH, e_ff_range, N_YSOS, Mass_Gass, Area_list, tff)

def main():
    """
    Main function to parse command-line arguments and run the SFE calculation pipeline.
    
    It retrieves simulation parameters and file paths from the command-line, handles 
    output directory creation, and then calls the main SFE_values function.
    """
    # Parse command-line options using docopt based on the module docstring
    options = docopt(__doc__)

    # Set simulation parameters from command-line options or use default values
    Box_Size = int(options["--BoxSize"]) if options["--BoxSize"] else 100
    R_I = int(options["--Starting_Range"]) if options["--Starting_Range"] else 65
    R_F = int(options["--Ending_Range"]) if options["--Ending_Range"] else 2500
    AGN = int(options["--AGN"]) if options["--AGN"] else 9  
    Distance = float(options["--Distance"]) if options["--Distance"] else 0.6

    # Handle output path
    if options["--output_path"]:
        OUTPATH = options["--output_path"]
        print(f"OUTPATH: {OUTPATH}")
        if isinstance(OUTPATH, str) and not os.path.isdir(OUTPATH):
            print(f"Creating directory: {OUTPATH}")
            os.mkdir(OUTPATH)
    else:
        OUTPATH = None

    # Retrieve file paths from the command-line arguments
    file1 = options["<file1>"]
    file2 = options["<file2>"]

    # Set simulation resolution from options or default to 1024
    RES = int(options["--Resolution"]) if options["--Resolution"] else 1024

    # If multiple files are provided, select the first one for each input
    if isinstance(file1, list):
        file1 = file1[0]
    if isinstance(file2, list):
        file2 = file2[0]

    # Run the SFE calculation pipeline with the provided parameters and file paths
    SFE_values(file1, file2, Box_Size, R_I, R_F, OUTPATH, RES)


if __name__ == "__main__":
    main()
