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
  --output_path=<PATH>       output path for data (defaults to /E_FF_range in file1 directory)
  --AGN                      set number of AGN contaminats per degree squared (default is 9)
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
from os import mkdir
from os.path import isdir

np.random.seed(42)


def read_yso_data(file_path):
    """
    Read YSO coordinates from the given file.
    If the file does not contain a "Masses" dataset, default to an array of zeros.
    """
    with h5py.File(file_path, "r") as f:
        YSO_X = f["X_pc"][:]
        YSO_Y = f["Y_pc"][:]
        if "Masses" in f:
            YSO_Mass = f["Masses"][:]
        else:
            # Use zeros if no mass data is available
            YSO_Mass = np.zeros(len(YSO_X))
    return YSO_X, YSO_Y, YSO_Mass


def read_dust_data(file_path):
    """
    Read dust coordinates and surface density from file,
    flatten the density array, and attach units.
    """
    with h5py.File(file_path, "r") as f:
        X = f["X_pc"][:]
        Y = f["Y_pc"][:]
        Surface_Den = f["SurfaceDensity_Msun_pc2"][:]
    new_den = Surface_Den.flatten() * (u.solMass / u.pc**2)
    dust_coords = np.c_[X.flatten(), Y.flatten()]
    return X, Y, Surface_Den, new_den, dust_coords


def build_dataframe(YSO_X, YSO_Y, YSO_Mass, new_den, dust_coords):
    """
    Use a KDTree to map each YSO coordinate to the nearest dust cell.
    Return a DataFrame with the YSO's X, Y positions, their corresponding dust
    surface density, and their masses.
    """
    targets = np.c_[YSO_X.flatten(), YSO_Y.flatten()]
    tree = KDTree(dust_coords)
    _, idx = tree.query(targets)
    df = pd.DataFrame({
        "YSO_X": targets[:, 0],
        "YSO_Y": targets[:, 1],
        "YSO Surface Den": new_den[idx],
        "YSO Mass": YSO_Mass
    })
    return df


def compute_sfe(df, new_den, Box_Size, R_I, R_F, RES):
    """
    Loop over density cutoff values and compute SFE (star formation efficiency)
    and various related metrics. Instead of storing results in separate lists,
    we build a list of dictionaries that is then converted to a Pandas DataFrame.
    """
    results_list = []
    # Convert gravitational constant to proper units
    new_G = G.to(u.pc**3 / (u.solMass * u.Myr**2)).value
    # Define a unit for surface density
    surface_unit = 1 * (u.solMass / u.pc**2)
    # Characteristic length from the simulation box
    L = Box_Size / 5

    # Use DataFrame columns for convenience
    yso_surface_den = df["YSO Surface Den"]
    yso_mass = df["YSO Mass"]

    # Loop through the density thresholds
    for i in np.arange(R_I, R_F, 1):
        threshold = i * surface_unit
        num_yso = (yso_surface_den > threshold).sum()
        StarRate = (((num_yso * 0.5) / 0.5) * (u.solMass / u.Myr))
        # Compute alternative SFR using average YSO mass above threshold
        if (yso_surface_den > threshold).sum() > 0:
            avg_mass = np.mean(yso_mass[yso_surface_den > threshold])
        else:
            avg_mass = 0
        no_assumption_SFR = (((num_yso * avg_mass) / 0.5) * (u.solMass / u.Myr))

        # Compute additional YSO mass statistics
        if (yso_surface_den > threshold).sum() > 0:
            mean_mass = np.mean(yso_mass[yso_surface_den > threshold])
            tot_mass = np.sum(yso_mass[yso_surface_den > threshold])
            med_mass = np.median(yso_mass[yso_surface_den > threshold])
        else:
            mean_mass = np.nan
            tot_mass = np.nan
            med_mass = np.nan

        # Compute gas mass and area from the dust data
        Mask = new_den > (i * surface_unit)
        A_i = (L / RES)**2 * u.pc**2
        A = (np.sum(Mask) * A_i).value
        M_Gas = np.sum(A_i * new_den[Mask]).value

        # Compute free-fall time and efficiency if possible
        if M_Gas <= 0 or A <= 0:
            T_ff = np.nan
            sfe = np.nan
            sfe_diff = np.nan
        else:
            t_ff_num = (A ** (3/2)) * np.sqrt(np.pi)
            t_ff_denom = 8 * new_G * M_Gas
            T_ff = np.sqrt(t_ff_num / t_ff_denom)
            sfe = (StarRate / (M_Gas / T_ff)).value
            sfe_diff = (no_assumption_SFR / (M_Gas / T_ff)).value

        results_list.append({
            "threshold": i,
            "SFE": sfe,
            "SFE_diff": sfe_diff,
            "N_YSOs": num_yso,
            "M_Gas": M_Gas,
            "Area": A,
            "T_ff": T_ff,
            "mean_mass": mean_mass,
            "total_mass": tot_mass,
            "median_mass": med_mass
        })
    results_df = pd.DataFrame(results_list)
    return results_df


def write_output(file1, OUTPATH, results_df):
    """
    Write the computed SFE values and associated metrics to an HDF5 file.
    This version uses h5py and creates separate datasets (as in your original code).
    The output filename is derived from the input YSO file name.
    """
    fname = os.path.basename(file1).replace(".YSOobjects.hdf5", ".e_ff_range.hdf5")
    if OUTPATH:
        outdir = OUTPATH
    else:
        outdir = os.path.join(str(pathlib.Path(file1).parent.resolve()), "E_FF")
    if not isdir(outdir):
        mkdir(outdir)
    imgpath = os.path.join(outdir, fname)

    with h5py.File(imgpath, "w") as F:
        # Create datasets from DataFrame columns
        F.create_dataset("SFE_Values", data=results_df["SFE"].to_numpy())
        F.create_dataset("SFE_Values_Different_Masses", data=results_df["SFE_diff"].to_numpy())
        F.create_dataset("NYSOs", data=results_df["N_YSOs"].to_numpy())
        F.create_dataset("M_Gas", data=results_df["M_Gas"].to_numpy())
        F.create_dataset("Area", data=results_df["Area"].to_numpy())
        F.create_dataset("T_ff", data=results_df["T_ff"].to_numpy())
        F.create_dataset("Average_YSO_Mass", data=results_df["mean_mass"].to_numpy())
        F.create_dataset("M_total", data=results_df["total_mass"].to_numpy())
        F.create_dataset("M_median", data=results_df["median_mass"].to_numpy())
    print("Output written to:", imgpath)


def SFE_values(file1, file2, Box_Size, R_I, R_F, OUTPATH, RES):
    # Read YSO and dust data from the input files
    YSO_X, YSO_Y, YSO_Mass = read_yso_data(file1)
    X, Y, Surface_Den, new_den, dust_coords = read_dust_data(file2)
    # Build a DataFrame mapping each YSO to the nearest dust surface density
    df = build_dataframe(YSO_X, YSO_Y, YSO_Mass, new_den, dust_coords)
    # Compute SFE values and associated metrics
    results_df = compute_sfe(df, new_den, Box_Size, R_I, R_F, RES)
    # Write the results to an output file using h5py datasets
    write_output(file1, OUTPATH, results_df)


def main():
    options = docopt(__doc__)

    Box_Size = int(options["--BoxSize"]) if options["--BoxSize"] else 100
    R_I = int(options["--Starting_Range"]) if options["--Starting_Range"] else 65
    R_F = int(options["--Ending_Range"]) if options["--Ending_Range"] else 2500

    if options["--output_path"]:
        OUTPATH = options["--output_path"]
        print(f"OUTPATH: {OUTPATH}")
        if isinstance(OUTPATH, str) and not isdir(OUTPATH):
            print(f"Creating directory: {OUTPATH}")
            mkdir(OUTPATH)
    else:
        OUTPATH = None

    file1 = options["<file1>"]
    file2 = options["<file2>"]

    RES = int(options["--Resolution"]) if options["--Resolution"] else 1024

    if isinstance(file1, list):
        file1 = file1[0]
    if isinstance(file2, list):
        file2 = file2[0]

    SFE_values(file1, file2, Box_Size, R_I, R_F, OUTPATH, RES)


if __name__ == "__main__":
    main()
