#! /bin/bash


HDF5_DIR="../snaps"

YSO_SCRIPT="../../../starforge_tools/mockobs/yso.py"
DUST_EMISSION="../../../starforge_tools/mockobs/dustemission.py"

for HDF5_FILE in "$HDF5_DIR"/*.hdf5; do

    echo "Loading YSO's"
    python3 "$YSO_SCRIPT" "$HDF5_FILE"
    echo "Creating Dust Map"
    python3 "$DUST_EMISSION" "$HDF5_FILE"

done 

