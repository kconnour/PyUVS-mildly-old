from pathlib import Path

from astropy.io import fits
from netCDF4 import Dataset
import numpy as np

import pyuvs as pu


def add_app_flip_to_nc_file(data_path: Path, nc_filename: Path):
    nc = Dataset(nc_filename, mode='r+', format='NETCDF4')
    orbit = pu.Orbit(int(nc['orbit'][:]))

    files = pu.find_latest_apoapse_muv_file_paths_from_block(data_path, orbit.orbit)
    hduls = [fits.open(f) for f in files]
    app = determine_app_orientation(hduls)

    app_flip = nc.createVariable('app_flip', 'i1')
    app_flip[:] = app


def determine_app_orientation(hduls: list) -> bool:
    vx = np.concatenate([f['spacecraftgeometry'].data['vx_instrument_inertial'][:, 0] for f in hduls])
    vsc = np.concatenate([f['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][:, 0] for f in hduls])
    dot = vx * vsc > 0
    return np.mean(dot) >= 0.5
