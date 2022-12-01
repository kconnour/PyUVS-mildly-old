from pathlib import Path
import typing

from astropy.io import fits
import h5py
import numpy as np

import pyuvs as pu

hdulist: typing.TypeAlias = fits.hdu.hdulist.HDUList


def make_hdf5_file_and_fill_with_fits_info(orbit: int, hdf5_save_location: Path, fits_files_location: Path) -> None:
    orbit = pu.Orbit(orbit)

    hdf5_filename = make_hdf5_filename(orbit, hdf5_save_location)
    make_empty_hdf5_file(hdf5_filename)

    fits_files = pu.find_latest_apoapse_muv_file_paths_from_block(fits_files_location, orbit.orbit)
    hduls = [fits.open(f) for f in fits_files]

    fill_hdf5_file_with_fits_info(orbit, hdf5_filename, hduls)


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


def fill_hdf5_file_with_fits_info(orbit: pu.Orbit, hdf5_filename: Path, fits_files: list[hdulist]) -> None:
    hdf5_file = h5py.File(hdf5_filename, mode='r+')  # 'r+' means read/write but the file must already exist
    create_hdf5_groups(hdf5_file)
    add_metadata(orbit, hdf5_file)

    # Add day/night independent data
    dayside_files = determine_dayside_files(fits_files)
    fill_integration_group(hdf5_file, fits_files, dayside_files)

    # Add day/night dependent data
    for dayside in [True, False]:
        daynight_hduls = [fits_files[c] for c, f in enumerate(dayside_files) if f == dayside]
        fill_detector_group(hdf5_file, daynight_hduls, dayside)
        fill_pixel_geometry_group(hdf5_file, daynight_hduls, dayside)
        fill_binning_group(hdf5_file, daynight_hduls, dayside)


def create_hdf5_groups(hdf5_file: h5py.File) -> None:
    hdf5_file.create_group('dayside_detector')
    hdf5_file.create_group('dayside_binning')
    hdf5_file.create_group('dayside_pixel_geometry')

    hdf5_file.create_group('nightside_detector')
    hdf5_file.create_group('nightside_binning')
    hdf5_file.create_group('nightside_pixel_geometry')

    hdf5_file.create_group('integration')


def add_metadata(orbit: pu.Orbit, hdf5_file: h5py.File) -> None:
    metadata = hdf5_file.create_dataset('metadata', data=h5py.Empty("i2"))

    metadata.attrs['Orbit'] = orbit.orbit
    metadata.attrs['IUVS_data_version'] = 13
    metadata.attrs['fits_to_hdf5_version'] = 1


def determine_dayside_files(fits_files: list[hdulist]) -> np.ndarray[bool]:
    return np.array([f['observation'].data['mcp_volt'][0] < pu.day_night_voltage_boundary for f in fits_files])


def fill_integration_group(hdf5_file: h5py.File, fits_files: list[hdulist], dayside_files: np.ndarray[bool]) -> None:
    integration = hdf5_file['integration']

    ephemeris_time_data = np.concatenate([f['integration'].data['et'] for f in fits_files])
    ephemeris_time = integration.create_dataset('ephemeris_time', data=ephemeris_time_data)
    ephemeris_time.attrs['unit'] = 'Seconds after J2000'

    field_of_view_data = np.concatenate([f['integration'].data['fov_deg'] for f in fits_files])
    field_of_view = integration.create_dataset('field_of_view', data=field_of_view_data)
    field_of_view.attrs['unit'] = 'Degrees'

    detector_temperature_data = np.concatenate([f['integration'].data['det_temp_c'] for f in fits_files])
    detector_temperature = integration.create_dataset('detector_temperature', data=detector_temperature_data)
    detector_temperature.attrs['unit'] = 'Degrees C'

    case_temperature_data = np.concatenate([f['integration'].data['case_temp_c'] for f in fits_files])
    case_temperature = integration.create_dataset('case_temperature', data=case_temperature_data)
    case_temperature.attrs['unit'] = 'Degrees C'

    sub_solar_latitude_data = np.concatenate([f['spacecraftgeometry'].data['sub_solar_lat'] for f in fits_files])
    sub_solar_latitude = integration.create_dataset('sub_solar_latitude', data=sub_solar_latitude_data)
    sub_solar_latitude.attrs['unit'] = 'Degrees [N]'

    sub_solar_longitude_data = np.concatenate([f['spacecraftgeometry'].data['sub_solar_lon'] for f in fits_files])
    sub_solar_longitude = integration.create_dataset('sub_solar_longitude', data=sub_solar_longitude_data)
    sub_solar_longitude.attrs['unit'] = 'Degrees [E]'

    sub_spacecraft_latitude_data = np.concatenate([f['spacecraftgeometry'].data['sub_spacecraft_lat'] for f in fits_files])
    sub_spacecraft_latitude = integration.create_dataset('sub_spacecraft_latitude', data=sub_spacecraft_latitude_data)
    sub_spacecraft_latitude.attrs['unit'] = 'Degrees [N]'

    sub_spacecraft_longitude_data = np.concatenate([f['spacecraftgeometry'].data['sub_spacecraft_lon'] for f in fits_files])
    sub_spacecraft_longitude = integration.create_dataset('sub_spacecraft_longitude', data=sub_spacecraft_longitude_data)
    sub_spacecraft_longitude.attrs['unit'] = 'Degrees [E]'

    spacecraft_altitude_data = np.concatenate([f['spacecraftgeometry'].data['spacecraft_alt'] for f in fits_files])
    spacecraft_altitude = integration.create_dataset('spacecraft_altitude', data=spacecraft_altitude_data)
    spacecraft_altitude.attrs['unit'] = 'km'

    # Make my own data
    n_integrations_per_file = [add_dimension_if_necessary(f['primary'].data, 3).shape[0] for f in fits_files]

    dayside_integrations_data = np.concatenate([np.repeat(dayside_files[f], n_integrations_per_file[f]) for f in range(len(fits_files))])
    dayside_integrations = integration.create_dataset('dayside_integrations', data=dayside_integrations_data)
    dayside_integrations.attrs['comment'] = 'True if dayside; False if nightside'

    voltage_data = np.concatenate([np.repeat(f['observation'].data['mcp_volt'], n_integrations_per_file[c]) for c, f in enumerate(fits_files)])
    voltage = integration.create_dataset('voltage', data=voltage_data)
    voltage.attrs['unit'] = 'V'

    voltage_gain_data = np.concatenate([np.repeat(f['observation'].data['mcp_gain'], n_integrations_per_file[c]) for c, f in enumerate(fits_files)])
    voltage_gain = integration.create_dataset('voltage_gain', data=voltage_gain_data)
    voltage_gain.attrs['unit'] = 'V'

    integration_time_data = np.concatenate([np.repeat(f['observation'].data['int_time'], n_integrations_per_file[c]) for c, f in enumerate(fits_files)])
    integration_time = integration.create_dataset('integration_time', data=integration_time_data)
    integration_time.attrs['unit'] = 'seconds'


def fill_detector_group(hdf5_file: h5py.File, daynight_hduls: list[hdulist], dayside: bool) -> None:
    detector = hdf5_file['dayside_detector'] if dayside else hdf5_file['nightside_detector']

    raw_data = np.vstack([add_dimension_if_necessary(f['detector_raw'].data, 3) for f in daynight_hduls])
    raw = detector.create_dataset('raw', data=raw_data)
    raw.attrs['unit'] = 'DN'

    dark_subtracted_data = np.vstack([add_dimension_if_necessary(f['detector_dark_subtracted'].data, 3) for f in daynight_hduls])
    dark_subtracted = detector.create_dataset('dark_subtracted', data=dark_subtracted_data)
    dark_subtracted.attrs['unit'] = 'DN'


def create_dataset(group: h5py.Group, data: np.ndarray, name: str, units: str) -> h5py.Dataset:
    dataset = group.create_dataset(name, data=data)
    dataset.attrs['unit'] = units
    return dataset


def fill_pixel_geometry_group(hdf5_file: h5py.File, daynight_hduls: list[hdulist], dayside: bool) -> None:
    pixel_geometry = hdf5_file['dayside_pixel_geometry'] if dayside else hdf5_file['nightside_pixel_geometry']

    # I thought this may reduce boilerplate code but imo it's not as readable. I hope to fix this if the structure of
    #  the file stays stable
    latitude = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_corner_lat'], 3) for f in daynight_hduls])
    longitude = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_corner_lon'], 3) for f in daynight_hduls])
    tangent_altitude = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_corner_mrh_alt'], 3) for f in daynight_hduls])
    local_time = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_local_time'], 2) for f in daynight_hduls])
    solar_zenith_angle = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_solar_zenith_angle'], 2) for f in daynight_hduls])
    emission_angle = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_emission_angle'], 2) for f in daynight_hduls])
    phase_angle = np.vstack([add_dimension_if_necessary(f['pixelgeometry'].data['pixel_phase_angle'], 2) for f in daynight_hduls])

    create_dataset(pixel_geometry, latitude, 'latitude', 'Degrees [N]')
    create_dataset(pixel_geometry, longitude, 'longitude', 'Degrees [E]')
    create_dataset(pixel_geometry, tangent_altitude, 'tangent_altitude', 'km')
    create_dataset(pixel_geometry, local_time, 'local_time', 'Hours')
    create_dataset(pixel_geometry, solar_zenith_angle, 'solar_zenith_angle', 'Degrees')
    create_dataset(pixel_geometry, emission_angle, 'emission_angle', 'Degrees')
    create_dataset(pixel_geometry, phase_angle, 'phase_angle', 'Degrees')


def fill_binning_group(hdf5_file: h5py.File, daynight_hduls: list[hdulist], dayside: bool) -> None:
    binning = hdf5_file['dayside_binning'] if dayside else hdf5_file['nightside_binning']

    spatial_pixel_edges_data = np.append(daynight_hduls[0]['binning'].data['spapixlo'][0], daynight_hduls[0]['binning'].data['spapixhi'][0, -1] + 1)
    spatial_pixel_edges = binning.create_dataset('spatial_pixel_edges', data=spatial_pixel_edges_data)
    spatial_pixel_edges.attrs['unit'] = 'Bin'
    spatial_pixel_edges.attrs['width'] = int(np.median(daynight_hduls[0]['binning'].data['spabinwidth'][0, 1:-1]))

    spectral_pixel_edges_data = np.append(daynight_hduls[0]['binning'].data['spepixlo'][0], daynight_hduls[0]['binning'].data['spepixhi'][0, -1] + 1)
    spectral_pixel_edges = binning.create_dataset('spectral_pixel_edges', data=spectral_pixel_edges_data)
    spectral_pixel_edges.attrs['unit'] = 'Bin'
    spectral_pixel_edges.attrs['width'] = int(np.median(daynight_hduls[0]['binning'].data['spebinwidth'][0, 1:-1]))


def add_dimension_if_necessary(array: np.ndarray, expected_dims: int) -> np.ndarray:
    return array if np.ndim(array) == expected_dims else array[None, :]


if __name__ == '__main__':
    # Example usage:
    hdf5_loc = Path('/media/kyle/iuvs/apoapse')
    fits_loc = Path('/media/kyle/iuvs/production')
    make_hdf5_file_and_fill_with_fits_info(4000, hdf5_loc, fits_loc)
