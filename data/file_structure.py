from pathlib import Path

from astropy.io import fits
import h5py
import numpy as np

import pyuvs as pu
from pipeline_versions import get_latest_pipeline_versions
from fits_to_hdf5 import add_dimension_if_necessary, determine_dayside_files
from file_name import make_empty_hdf5_file, make_hdf5_filename


def make_hdf5_file(orbit: int, hdf5_location: Path, fits_location: Path) -> None:
    orbit = pu.Orbit(orbit)
    hdf5_filename = make_hdf5_filename(orbit, hdf5_location)
    make_empty_hdf5_file(hdf5_filename)

    pipeline_versions = get_latest_pipeline_versions()

    hdf5_file = h5py.File(hdf5_filename, mode='r+')  # 'r+' means read/write but the file must already exist

    add_metadata(orbit, hdf5_file)
    create_groups(hdf5_file)
    add_version_info(hdf5_file, pipeline_versions)

    # Get data
    fits_files = pu.find_latest_apoapse_muv_file_paths_from_block(fits_location, orbit.orbit)
    hduls = [fits.open(f) for f in fits_files]

    # Make daynight independent stuff
    dayside_shape = determine_daynight_shape(hduls, True)
    nightside_shape = determine_daynight_shape(hduls, False)
    n_integrations = dayside_shape[0] + nightside_shape[0]

    create_empty_integration_datasets(hdf5_file, n_integrations)
    shapes = [nightside_shape, dayside_shape]

    # Make daynight dependent stuff
    for dayside in [True, False]:
        create_empty_detector_datasets(hdf5_file, shapes[dayside], dayside)
        create_empty_pixel_geometry_datasets(hdf5_file, shapes[dayside], dayside)
        create_empty_binning_datasets(hdf5_file, shapes[dayside], dayside)

    hdf5_file.close()


def determine_daynight_shape(hduls, dayside: bool) -> tuple[int, int, int]:
    daynight_files = determine_dayside_files(hduls) == dayside

    if np.any(daynight_files):
        daynight_hduls = [hduls[c] for c, f in enumerate(daynight_files) if f]
        detector_image = np.vstack([add_dimension_if_necessary(f['detector_raw'].data, 3) for f in daynight_hduls])
        detector_shape = detector_image.shape
    else:
        detector_shape = (0, 0, 0)
    return detector_shape


def add_metadata(orbit: pu.Orbit, hdf5_file: h5py.File) -> None:
    hdf5_file.attrs['orbit'] = orbit.orbit
    hdf5_file.attrs['app_flip'] = 0
    hdf5_file.attrs['n_swaths'] = 0


def add_version_info(hdf5_file: h5py.File, pipeline_version: dict) -> None:
    versions = hdf5_file['versions']
    # Set all keys to 0. They will be updated when part of a pipeline processes.
    for key in pipeline_version.keys():
        versions.attrs[key] = 0

    versions.attrs['structure'] = pipeline_version['structure']


def create_groups(hdf5_file: h5py.File) -> None:
    hdf5_file.create_group('versions')
    hdf5_file.create_group('integration')

    hdf5_file.create_group('dayside_detector')
    hdf5_file.create_group('dayside_binning')
    hdf5_file.create_group('dayside_pixel_geometry')

    hdf5_file.create_group('nightside_detector')
    hdf5_file.create_group('nightside_binning')
    hdf5_file.create_group('nightside_pixel_geometry')


def create_empty_integration_datasets(hdf5_file: h5py.File, n_integrations) -> None:
    integration = hdf5_file['integration']
    shape = (n_integrations,)

    ephemeris_time = integration.create_dataset('ephemeris_time', shape=shape, dtype='f8')
    ephemeris_time.attrs['unit'] = 'Seconds after J2000'

    field_of_view = integration.create_dataset('field_of_view', shape=shape, dtype='f4')
    field_of_view.attrs['unit'] = 'Degrees'

    detector_temperature = integration.create_dataset('detector_temperature', shape=shape, dtype='f8')
    detector_temperature.attrs['unit'] = 'Degrees C'

    case_temperature = integration.create_dataset('case_temperature', shape=shape, dtype='f8')
    case_temperature.attrs['unit'] = 'Degrees C'

    sub_solar_latitude = integration.create_dataset('sub_solar_latitude', shape=shape, dtype='f8')
    sub_solar_latitude.attrs['unit'] = 'Degrees [N]'

    sub_solar_longitude = integration.create_dataset('sub_solar_longitude', shape=shape, dtype='f8')
    sub_solar_longitude.attrs['unit'] = 'Degrees [E]'

    sub_spacecraft_latitude = integration.create_dataset('sub_spacecraft_latitude', shape=shape, dtype='f8')
    sub_spacecraft_latitude.attrs['unit'] = 'Degrees [N]'

    sub_spacecraft_longitude = integration.create_dataset('sub_spacecraft_longitude', shape=shape, dtype='f8')
    sub_spacecraft_longitude.attrs['unit'] = 'Degrees [E]'

    spacecraft_altitude = integration.create_dataset('spacecraft_altitude', shape=shape, dtype='f8')
    spacecraft_altitude.attrs['unit'] = 'km'

    dayside_integrations = integration.create_dataset('dayside_integrations', shape=shape, dtype='bool')
    dayside_integrations.attrs['comment'] = 'True if dayside; False if nightside'

    voltage = integration.create_dataset('voltage', shape=shape, dtype='f4')
    voltage.attrs['unit'] = 'V'

    voltage_gain = integration.create_dataset('voltage_gain', shape=shape, dtype='f8')
    voltage_gain.attrs['unit'] = 'V'

    integration_time = integration.create_dataset('integration_time', shape=shape, dtype='f8')
    integration_time.attrs['unit'] = 'seconds'

    swath_number = integration.create_dataset('swath_number', shape=shape, dtype='i2')
    swath_number.attrs['comment'] = 'The swath number corresponding to each integration'

    relay_integrations = integration.create_dataset('relay_integrations', shape=shape, dtype='bool')
    relay_integrations.attrs['comment'] = 'True if a relay integration; False otherwise (nominal data)'


def create_empty_detector_datasets(hdf5_file, shape, dayside: bool) -> None:
    detector = hdf5_file['dayside_detector'] if dayside else hdf5_file['nightside_detector']

    raw = detector.create_dataset('raw', shape=shape, dtype='f4')
    raw.attrs['unit'] = 'DN'

    dark_subtracted = detector.create_dataset('dark_subtracted', shape=shape, dtype='f8')
    dark_subtracted.attrs['unit'] = 'DN'

    brightness = detector.create_dataset('brightness', shape=shape, dtype='f8')
    brightness.attrs['unit'] = 'kR'

    if dayside:
        radiance = detector.create_dataset('radiance', shape=shape, dtype='f8')
        radiance.attrs['comment'] = 'Also known as I/F (unitless)'

        dust_opacity = detector.create_dataset('dust_opacity', shape=shape, dtype='f8')
        dust_opacity.attrs['unit'] = 'Column-integrated optical depth (unitless)'

        ice_opacity = detector.create_dataset('ice_opacity', shape=shape, dtype='f8')
        ice_opacity.attrs['unit'] = 'Column-integrated optical depth (unitless)'

        ozone_column = detector.create_dataset('ozone_column', shape=shape, dtype='f8')
        ozone_column.attrs['unit'] = 'Precipitable microns'

    else:
        co_cameron_bands = detector.create_dataset('co_cameron_bands', shape=shape, dtype='f8')
        co_cameron_bands.attrs['unit'] = 'kR'

        co_plus = detector.create_dataset('co+_first_negative', shape=shape, dtype='f8')
        co_plus.attrs['unit'] = 'kR'

        co2p_fdb = detector.create_dataset('co2+_fox_duffendack_barker', shape=shape, dtype='f8')
        co2p_fdb.attrs['unit'] = 'kR'

        co2_uvd = detector.create_dataset('co2+_ultraviolet_doublet', shape=shape, dtype='f8')
        co2_uvd.attrs['unit'] = 'kR'

        n2vk = detector.create_dataset('nitrogen2_vergard_kaplan', shape=shape, dtype='f8')
        n2vk.attrs['unit'] = 'kR'

        no_nightglow = detector.create_dataset('no_nightglow', shape=shape, dtype='f8')
        no_nightglow.attrs['unit'] = 'kR'

        oxygen_2792 = detector.create_dataset('oxygen_2792', shape=shape, dtype='f8')
        oxygen_2792.attrs['unit'] = 'kR'

        solar_continuum = detector.create_dataset('solar_continuum', shape=shape, dtype='f8')
        solar_continuum.attrs['unit'] = 'kR'


def create_empty_pixel_geometry_datasets(hdf5_file: h5py.File, shape: tuple, dayside: bool) -> None:
    pixel_geometry = hdf5_file['dayside_pixel_geometry'] if dayside else hdf5_file['nightside_pixel_geometry']
    pixel_shape = shape[:-1]
    pixel_corner_shape = pixel_shape + (5,)

    latitude = pixel_geometry.create_dataset('latitude', shape=pixel_corner_shape, dtype='f8')
    latitude.attrs['unit'] = 'Degrees [N]'

    longitude = pixel_geometry.create_dataset('longitude', shape=pixel_corner_shape, dtype='f8')
    longitude.attrs['unit'] = 'Degrees [E]'

    tangent_altitude = pixel_geometry.create_dataset('tangent_altitude', shape=pixel_corner_shape, dtype='f8')
    tangent_altitude.attrs['unit'] = 'km'

    local_time = pixel_geometry.create_dataset('local_time', shape=pixel_shape, dtype='f8')
    local_time.attrs['unit'] = 'km'

    solar_zenith_angle = pixel_geometry.create_dataset('solar_zenith_angle', shape=pixel_shape, dtype='f8')
    solar_zenith_angle.attrs['unit'] = 'Degrees'

    emission_angle = pixel_geometry.create_dataset('emission_angle', shape=pixel_shape, dtype='f8')
    emission_angle.attrs['unit'] = 'Degrees'

    phase_angle = pixel_geometry.create_dataset('phase_angle', shape=pixel_shape, dtype='f8')
    phase_angle.attrs['unit'] = 'Degrees'


def create_empty_binning_datasets(hdf5_file: h5py.File, shape: tuple, dayside: bool) -> None:
    binning = hdf5_file['dayside_binning'] if dayside else hdf5_file['nightside_binning']

    spatial_bin_edges = binning.create_dataset('spatial_bin_edges', shape=(shape[1] + 1,), dtype='i2')
    spatial_bin_edges.attrs['unit'] = 'Pixel number'
    spatial_bin_edges.attrs['width'] = 0

    spectral_pixel_edges = binning.create_dataset('spectral_bin_edges', shape=(shape[2] + 1,), dtype='i2')
    spectral_pixel_edges.attrs['unit'] = 'Pixel number'
    spectral_pixel_edges.attrs['width'] = 0


if __name__ == '__main__':
    hdf5_loc = Path('/media/kyle/iuvs/apoapse')
    fits_loc = Path('/media/kyle/iuvs/production')
    make_hdf5_file(4001, hdf5_loc, fits_loc)
