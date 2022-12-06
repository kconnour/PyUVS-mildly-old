from pathlib import Path
import typing

from astropy.io import fits
import h5py
import numpy as np

import pyuvs as pu
from file_name import make_hdf5_filename
from pipeline_versions import current_file_is_up_to_date, get_latest_pipeline_versions

hdulist: typing.TypeAlias = fits.hdu.hdulist.HDUList


def fill_hdf5_file_with_fits_info(orbit: int, hdf5_location: Path, fits_location: Path) -> None:
    orbit = pu.Orbit(orbit)
    hdf5_filename = make_hdf5_filename(orbit, hdf5_location)

    hdf5_file = h5py.File(hdf5_filename, mode='r+')  # 'r+' means read/write but the file must already exist
    update_data_file_versions(hdf5_file)

    fits_files = pu.find_latest_apoapse_muv_file_paths_from_block(fits_location, orbit.orbit)
    hduls = [fits.open(f) for f in fits_files]

    # Add day/night independent data
    dayside_files = determine_dayside_files(hduls)
    fill_integration_group(hdf5_file, hduls, dayside_files)

    # Add day/night dependent data
    for dayside in [True, False]:
        daynight_hduls = [hduls[c] for c, f in enumerate(dayside_files) if f == dayside]
        try:
            fill_detector_group(hdf5_file, daynight_hduls, dayside)
            fill_pixel_geometry_group(hdf5_file, daynight_hduls, dayside)
            fill_binning_group(hdf5_file, daynight_hduls, dayside)
        # This is the case if no daynight_hduls. np.vstack([]) gives an error
        except ValueError:
            continue

    hdf5_file.close()


def update_data_file_versions(hdf5_file: h5py.File) -> None:
    pipeline_versions = get_latest_pipeline_versions()

    hdf5_file['versions'].attrs['fits_to_hdf5'] = pipeline_versions['fits_to_hdf5']
    hdf5_file['versions'].attrs['IUVS_data'] = pipeline_versions['IUVS_data']


def determine_dayside_files(fits_files: list[hdulist]) -> np.ndarray[bool]:
    return np.array([f['observation'].data['mcp_volt'][0] < pu.day_night_voltage_boundary for f in fits_files])


def fill_integration_group(hdf5_file: h5py.File, fits_files: list[hdulist], dayside_files: np.ndarray[bool]) -> None:
    integration = hdf5_file['integration']

    integration['ephemeris_time'][:] = np.concatenate([f['integration'].data['et'] for f in fits_files])
    integration['field_of_view'][:] = np.concatenate([f['integration'].data['fov_deg'] for f in fits_files])
    integration['detector_temperature'][:] = np.concatenate([f['integration'].data['det_temp_c'] for f in fits_files])
    integration['case_temperature'][:] = np.concatenate([f['integration'].data['case_temp_c'] for f in fits_files])

    integration['sub_solar_latitude'][:] = np.concatenate([f['spacecraftgeometry'].data['sub_solar_lat'] for f in fits_files])
    integration['sub_solar_longitude'][:] = np.concatenate([f['spacecraftgeometry'].data['sub_solar_lon'] for f in fits_files])
    integration['sub_spacecraft_latitude'][:] = np.concatenate([f['spacecraftgeometry'].data['sub_spacecraft_lat'] for f in fits_files])
    integration['sub_spacecraft_longitude'][:] = np.concatenate([f['spacecraftgeometry'].data['sub_spacecraft_lon'] for f in fits_files])
    integration['spacecraft_altitude'][:] = np.concatenate([f['spacecraftgeometry'].data['spacecraft_alt'] for f in fits_files])

    # Make my own data
    n_integrations_per_file = [add_dimension_if_necessary(f['primary'].data, 3).shape[0] for f in fits_files]

    integration['dayside_integrations'][:] = np.concatenate([np.repeat(dayside_files[f], n_integrations_per_file[f]) for f in range(len(fits_files))])
    integration['voltage'][:] = np.concatenate([np.repeat(f['observation'].data['mcp_volt'], n_integrations_per_file[c]) for c, f in enumerate(fits_files)])
    integration['voltage_gain'][:] = np.concatenate([np.repeat(f['observation'].data['mcp_gain'], n_integrations_per_file[c]) for c, f in enumerate(fits_files)])
    integration['integration_time'][:] = np.concatenate([np.repeat(f['observation'].data['int_time'], n_integrations_per_file[c]) for c, f in enumerate(fits_files)])


def fill_detector_group(hdf5_file: h5py.File, daynight_hduls: list[hdulist], dayside: bool) -> None:
    detector = hdf5_file['dayside_detector'] if dayside else hdf5_file['nightside_detector']

    detector['raw'][:] = np.vstack([add_dimension_if_necessary(f['detector_raw'].data, 3) for f in daynight_hduls])
    detector['dark_subtracted'][:] = np.vstack([add_dimension_if_necessary(f['detector_dark_subtracted'].data, 3) for f in daynight_hduls])


def fill_pixel_geometry_group(hdf5_file: h5py.File, daynight_hduls: list[hdulist], dayside: bool) -> None:
    pixel_geometry = hdf5_file['dayside_pixel_geometry'] if dayside else hdf5_file['nightside_pixel_geometry']

    pixel_geometry['latitude'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_corner_lat'], 3) for f in daynight_hduls])
    pixel_geometry['longitude'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_corner_lon'], 3) for f in daynight_hduls])
    pixel_geometry['tangent_altitude'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_corner_mrh_alt'], 3) for f in daynight_hduls])
    pixel_geometry['local_time'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_local_time'], 2) for f in daynight_hduls])
    pixel_geometry['solar_zenith_angle'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_solar_zenith_angle'], 2) for f in daynight_hduls])
    pixel_geometry['emission_angle'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_emission_angle'], 2) for f in daynight_hduls])
    pixel_geometry['phase_angle'][:] = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_phase_angle'], 2) for f in daynight_hduls])


def fill_binning_group(hdf5_file: h5py.File, daynight_hduls: list[hdulist], dayside: bool) -> None:
    binning = hdf5_file['dayside_binning'] if dayside else hdf5_file['nightside_binning']

    spatial_bin_edges = binning['spatial_bin_edges']
    spatial_bin_edges[:] = np.append(daynight_hduls[0]['binning'].data['spapixlo'][0], daynight_hduls[0]['binning'].data['spapixhi'][0, -1] + 1)
    spatial_bin_edges.attrs['width'] = int(np.median(daynight_hduls[0]['binning'].data['spabinwidth'][0, 1:-1]))

    spectral_bin_edges = binning['spectral_bin_edges']
    spectral_bin_edges[:] = np.append(daynight_hduls[0]['binning'].data['spepixlo'][0], daynight_hduls[0]['binning'].data['spepixhi'][0, -1] + 1)
    spectral_bin_edges.attrs['width'] = int(np.median(daynight_hduls[0]['binning'].data['spebinwidth'][0, 1:-1]))


def add_dimension_if_necessary(array: np.ndarray, expected_dims: int) -> np.ndarray:
    return array if np.ndim(array) == expected_dims else array[None, :]


if __name__ == '__main__':
    # Example usage:
    hdf5_loc = Path('/media/kyle/iuvs/apoapse')
    fits_loc = Path('/media/kyle/iuvs/production')

    hdf5_file = h5py.File('/media/kyle/iuvs/apoapse/orbit04000/apoapse-orbit04002-muv-new.hdf5', mode='r+')

    if not current_file_is_up_to_date(hdf5_file, 'fits_to_hdf5'):
        print('doing')
        fill_hdf5_file_with_fits_info(4002, hdf5_loc, fits_loc)

    sza = hdf5_file['nightside_pixel_geometry']['solar_zenith_angle']
    print(sza[:])
    print(type(sza[:]), sza.shape)