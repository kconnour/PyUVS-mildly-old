from pathlib import Path

from astropy.io import fits
from netCDF4 import Dataset
import numpy as np

import pyuvs as pu


def create_nc_from_fits(orbit: pu.Orbit, data_location: Path, save_location: Path) -> None:
    """Create a netCDF4 file from fits files of IUVS data.

    Parameters
    ----------
    orbit
        The orbit number
    data_location
        The location where to find the fits files.
    save_location
        The location where to save the nc files.

    Returns
    -------
    None
    """
    files = pu.find_latest_apoapse_muv_file_paths_from_block(data_location, orbit.orbit)
    hduls = [fits.open(f) for f in files]
    dayside_files = np.array([f['observation'].data['mcp_volt'][0] < pu.day_night_voltage_boundary for f in hduls])
    for dayside in [True, False]:
        daynight_files = dayside_files == dayside
        if not np.any(daynight_files):
            continue

        daynight_hduls = [hduls[c] for c, f in enumerate(daynight_files) if f]
        nc = make_daynight_nc_structure(orbit, daynight_hduls, save_location, dayside)
        fill_nc_file_with_hdul_info(nc, daynight_hduls)


def make_daynight_nc_structure(orbit: pu.Orbit, hduls: list, save_location: Path, dayside: bool) -> Dataset:
    # Make the .nc file on the system
    nc_filename = make_nc_filename(orbit, save_location, dayside)
    nc_filename.parent.mkdir(parents=True, exist_ok=True)
    nc = Dataset(nc_filename, 'w', format='NETCDF4')

    # Make the .nc dimensions
    integrations, spatial_bins, spectral_bins = get_detector_image_shape(hduls)
    nc.createDimension('integrations', size=integrations)
    nc.createDimension('spatial_bins', size=spatial_bins)
    nc.createDimension('spatial_bin_edges', size=spatial_bins + 1)
    nc.createDimension('spectral_bins', size=spectral_bins)
    nc.createDimension('spectral_bin_edges', size=spectral_bins + 1)
    nc.createDimension('pixel_corners', size=5)

    # Make the .nc structure
    nc.createGroup('apoapse')
    nc.createGroup('binning')
    nc.createGroup('detector')
    nc.createGroup('integration')
    nc.createGroup('observation')
    nc.createGroup('pixel_geometry')
    nc.createGroup('spacecraft_geometry')

    return nc


def make_nc_filename(orbit: pu.Orbit, save_location: Path, dayside: bool) -> Path:
    save_directory = save_location / orbit.block
    filename = f'apoapse-{orbit.code}-muv-dayside.nc' if dayside else f'apoapse-{orbit.code}-muv-nightside.nc'
    return save_directory / filename


def get_detector_image_shape(hduls: list) -> tuple[int, int, int]:
    stacked_integrations = np.vstack([add_dimension_if_necessary(f['primary'].data, 3) for c, f in enumerate(hduls)])
    return stacked_integrations.shape


def add_dimension_if_necessary(array: np.ndarray, expected_dims: int) -> np.ndarray:
    return array if np.ndim(array) == expected_dims else array[None, :]


def fill_nc_file_with_hdul_info(nc: Dataset, hduls: list) -> None:
    fill_detector_group(nc, hduls)
    fill_pixelgeometry_group(nc, hduls)
    fill_binning_group(nc, hduls)
    fill_observation_group(nc, hduls)
    fill_integration_group(nc, hduls)


def fill_detector_group(nc: Dataset, hduls: list) -> None:
    detector = nc.groups['detector']
    detector_shape = ('integrations', 'spatial_bins', 'spectral_bins')

    raw = detector.createVariable('raw', 'f4', detector_shape)
    raw.units = 'DN'
    raw[:] = np.vstack([add_dimension_if_necessary(f['detector_raw'].data, 3) for f in hduls])

    dark_subtracted = detector.createVariable('dark_subtracted', 'f4', detector_shape)
    dark_subtracted.units = 'DN'
    dark_subtracted[:] = np.vstack([add_dimension_if_necessary(f['detector_dark_subtracted'].data, 3) for f in hduls])


def fill_pixelgeometry_group(nc, hduls: list) -> None:
    pixel_geometry = nc.groups['pixel_geometry']
    pixel_corner_shape = ('integrations', 'spatial_bins', 'pixel_corners')
    pixel_shape = ('integrations', 'spatial_bins')

    latitude = pixel_geometry.createVariable('latitude', 'f8', pixel_corner_shape)
    latitude.units = 'Degrees [N]'
    latitude[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_corner_lat'], 3) for f in hduls])

    longitude = pixel_geometry.createVariable('longitude', 'f8', pixel_corner_shape)
    longitude.units = 'Degrees [E]'
    longitude[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_corner_lon'], 3) for f in hduls])

    tangent_altitude = pixel_geometry.createVariable('tangent_altitude', 'f8', pixel_corner_shape)
    tangent_altitude.units = 'km'
    tangent_altitude[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_corner_mrh_alt'], 3) for f in hduls])

    local_time = pixel_geometry.createVariable('local_time', 'f8', pixel_shape)
    local_time.units = 'Hours'
    local_time[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_local_time'], 2) for f in hduls])

    solar_zenith_angle = pixel_geometry.createVariable('solar_zenith_angle', 'f8', pixel_shape)
    solar_zenith_angle.units = 'Degrees'
    solar_zenith_angle[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_solar_zenith_angle'], 2) for f in hduls])

    emission_angle = pixel_geometry.createVariable('emission_angle', 'f8', pixel_shape)
    emission_angle.units = 'Degrees'
    emission_angle[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_emission_angle'], 2) for f in hduls])

    phase_angle = pixel_geometry.createVariable('phase_angle', 'f8', pixel_shape)
    phase_angle.units = 'Degrees'
    phase_angle[:] = np.vstack([add_dimension_if_necessary(
        f['pixelgeometry'].data['pixel_phase_angle'], 2) for f in hduls])


def fill_binning_group(nc: Dataset, hduls: list) -> None:
    binning = nc.groups['binning']

    spatial_pixel_edges = binning.createVariable('spatial_pixel_edges', 'i2', ('spatial_bin_edges',))
    spatial_pixel_edges.units = 'Bin'
    spatial_pixel_edges[:] = np.append(
        hduls[0]['binning'].data['spapixlo'][0], hduls[0]['binning'].data['spapixhi'][0, -1]+1)

    spectral_pixel_edges = binning.createVariable('spectral_pixel_edges', 'i2', ('spectral_bin_edges',))
    spectral_pixel_edges.units = 'Bin'
    spectral_pixel_edges[:] = np.append(
        hduls[0]['binning'].data['spepixlo'][0], hduls[0]['binning'].data['spepixhi'][0, -1]+1)

    spatial_bin_width = binning.createVariable('spatial_bin_width', 'i2')
    spatial_bin_width.units = 'Bin'
    spatial_bin_width[:] = int(np.median(hduls[0]['binning'].data['spabinwidth'][0, 1:-1]))

    spectral_bin_width = binning.createVariable('spectral_bin_width', 'i2')
    spectral_bin_width.units = 'Bin'
    spectral_bin_width[:] = int(np.median(hduls[0]['binning'].data['spebinwidth'][0, 1:-1]))


def fill_observation_group(nc, hduls: list) -> None:
    observation = nc.groups['observation']

    integration_time = observation.createVariable('integration_time', 'i2')
    integration_time.units = 'Seconds'
    integration_time[:] = hduls[0]['observation'].data['int_time']

    voltage = observation.createVariable('voltage', 'f8')
    voltage.units = 'Volts'
    voltage[:] = hduls[0]['observation'].data['mcp_volt']

    voltage_gain = observation.createVariable('voltage_gain', 'f8')
    voltage_gain.units = ''
    voltage_gain[:] = hduls[0]['observation'].data['mcp_gain']


def fill_integration_group(nc, hduls: list) -> None:
    integration = nc.groups['integration']
    integration_shape = ('integrations',)

    sub_solar_latitude = integration.createVariable('sub_solar_latitude', 'f8', integration_shape)
    sub_solar_latitude.units = 'Degrees'
    sub_solar_latitude[:] = np.concatenate([f['spacecraftgeometry'].data['sub_solar_lat'] for f in hduls])

    sub_solar_longitude = integration.createVariable('sub_solar_longitude', 'f8', integration_shape)
    sub_solar_longitude.units = 'Degrees'
    sub_solar_longitude[:] = np.concatenate([f['spacecraftgeometry'].data['sub_solar_lon'] for f in hduls])

    ephemeris_time = integration.createVariable('ephemeris_time', 'f8', integration_shape)
    ephemeris_time.units = 'Seconds after J2000'
    ephemeris_time[:] = np.concatenate([f['integration'].data['et'] for f in hduls])

    field_of_view = integration.createVariable('field_of_view', 'f8', integration_shape)
    field_of_view.units = 'Degrees'
    field_of_view[:] = np.concatenate([f['integration'].data['fov_deg'] for f in hduls])


if __name__ == '__main__':
    '''import multiprocessing as mp
    import time
    t0 = time.time()
    p = Path('/media/kyle/McDataFace/iuvsdata/production')
    saveloc = Path('/home/kyle/iuvs/apoapse')

    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus - 1)

    for o in range(4000, 4100):
        pool.apply_async(func=create_nc_from_fits, args=(pu.Orbit(o), Path('/media/kyle/McDataFace/iuvsdata/production'), Path('/home/kyle/iuvs/apoapse')))
    pool.close()
    pool.join()
    print(f'The data files took {time.time() - t0} seconds to process.')'''

