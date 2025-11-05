#!/usr/bin/env python
"""
Generates a range of values of star formation efficiency from the time-averaged
accretion rates at a single snapshot in each contour.

Usage: e_ff.py <file1> <file2> <file3> [options]

Options:
  -h --help                  Show this screen
  --BoxSize=<BX>             Box size of the simulation (defaults to 100)
  --Starting_Range=<R_i>     Starting range of density cutoff (defaults to 65)
  --Ending_Range=<R_f>       Ending range of density cutoff (defaults to 2500)
  --Resolution=<R>           Resolution of simulation (defaults to 1024)
  --output_path=<PATH>       Output path (defaults to E_FF under file1’s dir)

file1: original snapshot file (with particle IDs)    # also read Header Time
file2: Dust-emission / mock-obs file    # same snapshot as file1
file3: Accretion-histories HDF5
"""

import os
from os import mkdir
from os.path import isdir
import pathlib

from scipy.spatial import cKDTree
from astropy.constants import G
import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from docopt import docopt

np.random.seed(42)

def read_dust_data(file_path):
    with h5py.File(file_path, "r") as f:
        X           = f["X_pc"][:]
        Y           = f["Y_pc"][:]
        Surface_Den = f["SurfaceDensity_Msun_pc2"][:]
    new_den     = Surface_Den.flatten() * (u.solMass/u.pc**2)
    dust_coords = np.c_[X.flatten(), Y.flatten()]
    return X, Y, Surface_Den, new_den, dust_coords


def read_snapshot_data(file_path, snapshot_mock_ob):
    with h5py.File(file_path, "r") as F:
        t_code   = F["Header"].attrs["Time"]
        t_snap   = float(t_code * 978.5)           # Myr
        gas_pos  = F["PartType0/Coordinates"][:]  # (Ngas,3) in pc
        gas_mass = F["PartType0/Masses"][:]       # (Ngas,) in M_sun
        gas_rho  = F["PartType0/Density"][:]      # (Ngas,) in M_sun/pc^3
        star_pos = F["PartType5/Coordinates"][:]  # (Nstar,3)
        star_ids = F["PartType5/ParticleIDs"][:]
        star_age = F["PartType5/ProtoStellarAge"][:] * 978.5  # (Nstar,)
        star_masses = F["PartType5/Masses"][:]

    _, _, Surface_den, flattened_surface, gas_xy = read_dust_data(snapshot_mock_ob)

    tree = cKDTree(gas_xy)
    star_targets = star_pos[:, :2]
    gas_targets  = gas_pos[:, :2]

    _, idx_stars = tree.query(star_targets)
    _, idx_gas    = tree.query(gas_targets)

    Star_Surface_Den = (flattened_surface[idx_stars]).value
    Gas_Surface_Den  = (flattened_surface[idx_gas]).value

    df_gas = {
        'Time Snap': t_snap,
        'Gas Positions': gas_pos,
        'Gas Masses': gas_mass,
        'Gas Density': gas_rho,
        'Gas Surface Density': Gas_Surface_Den
    }
    df_stars = {
        'Star Positions': star_pos,
        'Star IDs': star_ids,
        'Star Ages': star_age,
        'Star Masses': star_masses,
        'Star Surface Density': Star_Surface_Den
    }
    # We'll convert to numpy arrays later, no need for pandas here
    return df_gas, df_stars


def load_accretion_rates(acc_file, t_snap):
    rates = {}
    with h5py.File(acc_file, "r") as f_acc:
        for grp in f_acc:
            if not grp.startswith("Star"):
                continue
            sid   = int(grp.replace("Star",""))
            t_arr = f_acc[grp]["Time_Myr"][:]     # Myr
            m_arr = f_acc[grp]["Mass_Msun"][:]    # M_sun
            if len(m_arr) < 2:
                rate = 0.0
            elif len(m_arr) == 2:
                dt = t_arr[1] - t_arr[0]
                dm = m_arr[1] - m_arr[0]
                rate = float(dm/dt)
            else:
                mdot_arr = np.gradient(m_arr, t_arr)
                idx = np.argmin(np.abs(t_arr - t_snap))
                rate = float(mdot_arr[idx])
            rates[sid] = rate
    return rates


def compute_sfe_fast(df_gas, df_stars, Box_Size, R_I, R_F, RES, acc_rates):
    # Convert dicts to arrays
    gas_dens = np.array(df_gas['Gas Surface Density'])
    gas_mass = np.array(df_gas['Gas Masses'])
    rho      = np.array(df_gas['Gas Density'])
    star_dens= np.array(df_stars['Star Surface Density'])
    star_ids = np.array(df_stars['Star IDs'])

    # free-fall time per gas cell
    G_pc = G.to(u.pc**3/(u.solMass*u.Myr**2)).value
    t_ff = np.sqrt(3*np.pi / (32 * G_pc * rho))
    gas_contrib = gas_mass / t_ff

    # mdot per star
    mdot_arr = np.array([acc_rates.get(int(sid), 0.0) for sid in star_ids])

    # sort ascending
    gas_order  = np.argsort(gas_dens)
    star_order = np.argsort(star_dens)
    gas_sorted   = gas_dens[gas_order]
    star_sorted  = star_dens[star_order]

    # cumulative sums high->low
    gas_cumsum  = np.cumsum(gas_contrib[gas_order])[::-1]
    star_cumsum = np.cumsum(mdot_arr[star_order])[::-1]

    thresholds = np.arange(R_I, R_F + 1)
    # find index where density falls below threshold
    idx_g = np.searchsorted(gas_sorted, thresholds, side='left')
    idx_s = np.searchsorted(star_sorted, thresholds, side='left')

    gas_term = gas_cumsum[idx_g]
    mdot_tot = star_cumsum[idx_s]

    eps_ff   = mdot_tot / gas_term
    eps_ff[gas_term == 0] = np.nan

    return pd.DataFrame({
        'Sigma_threshold': thresholds,
        'Mdot_tot_Msun_per_Myr': mdot_tot,
        'sum_Mgas_over_tff': gas_term,
        'epsilon_true': eps_ff,
    })


def write_output(file1, OUTPATH, df):
    fname  = os.path.basename(file1).replace('.hdf5', '.e_ff_accretion.theory.hdf5')
    outdir = OUTPATH if OUTPATH else os.path.join(pathlib.Path(file1).parent.resolve(), 'E_FF')
    if not isdir(outdir):
        mkdir(outdir)
    outpath = os.path.join(outdir, fname)
    with h5py.File(outpath, 'w') as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].to_numpy())
    print('Wrote:', outpath)


def SFE_values(file1, file2, file3, Box_Size, R_I, R_F, OUTPATH, RES):
    df_gas_dict, df_stars_dict = read_snapshot_data(file1, file2)
    acc_rates = load_accretion_rates(file3, df_gas_dict['Time Snap'])
    results_df = compute_sfe_fast(df_gas_dict, df_stars_dict, Box_Size, R_I, R_F, RES, acc_rates)
    write_output(file1, OUTPATH, results_df)


def main():
    opts     = docopt(__doc__)
    Box_Size = int(opts['--BoxSize'] or 100)
    R_I      = int(opts['--Starting_Range'] or 65)
    R_F      = int(opts['--Ending_Range'] or 2500)
    RES      = int(opts['--Resolution'] or 1024)
    OUTPATH  = opts['--output_path']
    file1, file2, file3 = opts['<file1>'], opts['<file2>'], opts['<file3>']
    SFE_values(file1, file2, file3, Box_Size, R_I, R_F, OUTPATH, RES)

if __name__ == '__main__':
    main()
