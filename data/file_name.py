from pathlib import Path

import h5py

import pyuvs as pu


def make_hdf5_filename(orbit: pu.Orbit, save_location: Path) -> Path:
    filename = f'apoapse-{orbit.code}-muv.hdf5'
    return save_location / orbit.block / filename


def make_empty_hdf5_file(hdf5_filename: Path) -> None:
    hdf5_filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        f = h5py.File(hdf5_filename, mode='x')  # 'x' means to create the file but fail if it already exists
        f.close()
    except FileExistsError:
        pass
