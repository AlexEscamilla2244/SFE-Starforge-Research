l#!/usr/bin/env python
"""
Generates a list of coordinates, luminosities, and masses of  
young stellar objects (YSOs) from simulation snapshots.

Usage:
  yso.py [options] <files>...

Options:
  -h --help                 Show this help message.
  --age=<tau>               Desired YSO age cutoff in Myr [default: 0.5]
  --output_path=<path>      Directory to write all YSO files (defaults to
                            a `YSOobjects/` subdir next to each snapshot)
  --num_jobs=<n>            Number of snapshots to process in parallel [default: 1]
"""
from docopt import docopt
from pathlib import Path
import h5py
import numpy as np
import astropy.units as u
from joblib import Parallel, delayed

# parse command line
opts = docopt(__doc__)
TAU        = float(opts["--age"])
NUM_JOBS   = int(opts["--num_jobs"])
OUT_PATH   = opts["--output_path"]  # None or string
FILES      = opts["<files>"]

# unit conversion factor: simulation time → Myr
conversion_time = (1 * u.parsec / (u.meter / u.second)).to(u.Myr).value

def process_snapshot(fn):
    fn = Path(fn)
    with h5py.File(fn, "r") as f:
        sim_time = f["Header"].attrs["Time"] * conversion_time
        coords   = f["PartType5/Coordinates"][:]      # shape (N,3)
        lum      = f["PartType5/StarLuminosity_Solar"][:]
        masses   = f["PartType5/Masses"][:]
        ages     = f["PartType5/ProtoStellarAge"][:] * conversion_time
        starID = f["PartType5/ParticleIDs"][:]

    # pick only stars younger than TAU
    mask = (sim_time - ages) <= TAU
    X, Y = coords[mask,0], coords[mask,1]
    L, M = lum[mask], masses[mask]
    ID = starID[mask]
    Star_Age = ages[mask]

    # decide where to write
    if OUT_PATH:
        outdir = Path(OUT_PATH)
    else:
        outdir = fn.parent / "YSOobjects"
    outdir.mkdir(parents=True, exist_ok=True)

    outfn = outdir / (fn.stem + ".YSOobjects.hdf5")
    with h5py.File(outfn, "w") as outf:
        outf.create_dataset("X_pc",       data=X)
        outf.create_dataset("Y_pc",       data=Y)
        outf.create_dataset("Luminosity", data=L)
        outf.create_dataset("Masses",     data=M)
        outf.create_dataset("StarIDs",    data=ID)
        outf.create_dataset("StarAge", data=Star_Age)

    print(f"WROTE {outfn}  (found {mask.sum()} YSOs)")

def main():
    Parallel(n_jobs=NUM_JOBS)(
        delayed(process_snapshot)(str(f)) for f in FILES
    )

if __name__ == "__main__":
    main()
