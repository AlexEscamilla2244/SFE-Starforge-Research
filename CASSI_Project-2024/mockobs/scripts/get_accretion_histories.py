#!/usr/bin/python

"""
Generates accretion rate histories for each individual star (PartType5 particles) and the cluster’s total accretion rate.

Usage:
  get_accretion_histories.py <path> [--output_path=<PATH>]

Options:
  -h --help             Show this screen.
  --output_path=<PATH>  Output directory for new files (defaults to each run’s "accretion_histories").
"""
import os
import h5py
import numpy as np
from glob import glob
from docopt import docopt

mmin = 0.1
mmax = 100.0

def calculate_accretion_histories(runs, OUTPATH):
    for run in runs:
        mdict = {}
        snapshot_times = []
        total_sink_mass = []

        files = sorted(glob(os.path.join(run, "snapshot*.hdf5")))
        for f in files:
            print(f)
            with h5py.File(f, "r") as F:
                if 'PartType5' not in F:
                    continue
                t_code = F['Header'].attrs['Time']
                t_myr  = t_code * 978.5
                snapshot_times.append(t_myr)
                msink_arr = F['PartType5/Masses'][:]
                total_sink_mass.append(msink_arr.sum())

                ids    = F['PartType5/ParticleIDs'][:]
                mstar  = F['PartType5/BH_Mass'][:]
                for pid, m, ms in zip(ids, mstar, msink_arr):
                    mdict.setdefault(int(pid), []).append([t_code, m, ms])

        odir = OUTPATH if OUTPATH else os.path.join(run, 'accretion_histories')
        os.makedirs(odir, exist_ok=True)
        outfile = os.path.join(odir, 'accretion_histories.hdf5')

        with h5py.File(outfile, "w") as F_acc:
            # write individual star histories
            for pid in sorted(mdict):
                records = np.array(mdict[pid])
                records = records[records[:, 0].argsort()]
                t_arr, m_arr, ms_arr = records.T
                m_zams = m_arr.max()
                if m_zams < mmin or m_zams > mmax:
                    continue
                grp = F_acc.create_group(f"Star{pid}")
                grp.create_dataset('ZAMS_Mass',      data=m_zams)
                grp.create_dataset('Time_Myr',       data=t_arr * 978.5)
                grp.create_dataset('Mass_Msun',      data=m_arr)
                grp.create_dataset('Mass_Sink_Msun', data=ms_arr)

            # compute and write cluster-wide accretion history
            times  = np.array(snapshot_times)
            masses = np.array(total_sink_mass)
            order  = np.argsort(times)
            times  = times[order]
            masses = masses[order]
            mdot   = np.diff(masses) / np.diff(times)
            mdot_full = np.concatenate(([np.nan], mdot))
            grp_c = F_acc.create_group('Cluster')
            grp_c.create_dataset('Time_Myr',       data=times)
            grp_c.create_dataset('Total_SinkMass', data=masses)
            grp_c.create_dataset('Total_Mdot',     data=mdot_full)

        print(f"Wrote {outfile}")

def main():
    args    = docopt(__doc__)
    path    = args['<path>']
    outpath = args['--output_path']
    if outpath and not os.path.isdir(outpath):
        os.makedirs(outpath)
    calculate_accretion_histories([path], outpath)

if __name__ == '__main__':
    main()
