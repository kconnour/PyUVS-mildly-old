from pathlib import Path
import typing

from astropy.io import fits
import h5py
import numpy as np

import pyuvs as pu
from pipeline_versions import current_file_is_up_to_date, get_latest_pipeline_versions
from file_name import make_hdf5_filename

hdulist: typing.TypeAlias = fits.hdu.hdulist.HDUList


def add_app_flip_to_hdf5_file(orbit: int, hdf5_location: Path, fits_files_location: Path) -> None:
    orbit = pu.Orbit(orbit)
    hdf5_filename = make_hdf5_filename(orbit, hdf5_location)
    hdf5_file = h5py.File(hdf5_filename, mode='r+')

    if not current_file_is_up_to_date(hdf5_file, 'APP_flip'):
        files = pu.find_latest_apoapse_muv_file_paths_from_block(fits_files_location, orbit.orbit)
        hduls = [fits.open(f) for f in files]
        app = determine_app_orientation(hduls)
        hdf5_file.attrs['app_flip'] = app
        update_data_file_versions(hdf5_file)


def determine_app_orientation(hduls: list[hdulist]) -> bool:
    vx = np.concatenate([f['spacecraftgeometry'].data['vx_instrument_inertial'][:, 0] for f in hduls])
    vsc = np.concatenate([f['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][:, 0] for f in hduls])
    dot = vx * vsc > 0
    return np.mean(dot) >= 0.5


def update_data_file_versions(hdf5_file: h5py.File) -> None:
    pipeline_versions = get_latest_pipeline_versions()

    hdf5_file['versions'].attrs['APP_flip'] = pipeline_versions['APP_flip']
