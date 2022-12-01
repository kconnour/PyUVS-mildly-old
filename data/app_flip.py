from pathlib import Path
import typing

from astropy.io import fits
import h5py
import numpy as np

import pyuvs as pu

hdulist: typing.TypeAlias = fits.hdu.hdulist.HDUList


def add_app_flip_to_hdf5_file(orbit: pu.Orbit, hdf5_filename: Path, fits_files_location: Path) -> None:
    hdf5_file = h5py.File(hdf5_filename, mode='r+')

    files = pu.find_latest_apoapse_muv_file_paths_from_block(fits_files_location, orbit.orbit)
    hduls = [fits.open(f) for f in files]
    app = determine_app_orientation(hduls)

    metadata = hdf5_file['metadata']
    metadata.attrs['APP_flip'] = app


def determine_app_orientation(hduls: list[hdulist]) -> bool:
    vx = np.concatenate([f['spacecraftgeometry'].data['vx_instrument_inertial'][:, 0] for f in hduls])
    vsc = np.concatenate([f['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][:, 0] for f in hduls])
    dot = vx * vsc > 0
    return np.mean(dot) >= 0.5


if __name__ == '__main__':
    # Example usage:
    from fits_to_hdf5 import make_hdf5_filename
    hdf5_loc = Path('/media/kyle/iuvs/apoapse')
    fits_loc = Path('/media/kyle/iuvs/production')

    hdf5_fname = make_hdf5_filename(pu.Orbit(4000), hdf5_loc)
    add_app_flip_to_hdf5_file(pu.Orbit(4000), hdf5_fname, fits_loc)
    h = h5py.File(hdf5_fname)
    md = h['metadata']
    app = md.attrs['APP_flip']
    print(app, type(app))
