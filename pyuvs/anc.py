"""This module provides functions to load in standard dictionaries and arrays
for working with IUVS data."""
from pathlib import Path
import numpy as np


_error_message = 'The given file was not downloaded before pyuvs instllation.'


def _get_package_path() -> Path:
    return Path(__file__).parent.resolve()


def _get_anc_directory() -> Path:
    return _get_package_path() / 'anc'


def _get_maps_directory() -> Path:
    return _get_anc_directory() / 'maps'


def _get_templates_directory() -> Path:
    return _get_anc_directory() / 'templates'


def _get_instrument_directory() -> Path:
    return _get_anc_directory() / 'instrument'


def load_map_magnetic_field_closed_probability() -> np.ndarray:
    """Load the map denoting the probability of a closed magnetic field line.

    Returns
    -------
    np.ndarray
        Array of the image.

    Notes
    -----
    This map comes from `MGS data <https://doi.org/10.1029/2007JA012435>`_. It
    has a shape of (180, 360).

    * The zeroth axis corresponds to latitude and spans -90 to 90 degrees.
    * The first axis corresponds to east longitude and spans 0 to 360 degrees.

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import matplotlib.ticker as ticker
       import pyuvs as pu

       fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

       b_field = pu.load_map_magnetic_field_closed_probability()
       ax.imshow(b_field, cmap='Blues_r', extent=[0, 360, -90, 90],
                          origin='lower', rasterized=True)
       ax.set_xlabel('Longitude [degrees]')
       ax.set_ylabel('Latitude [degrees]')
       ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
       ax.yaxis.set_major_locator(ticker.MultipleLocator(30))

       plt.show()

    """
    try:
        return np.load(_get_maps_directory() / 'magnetic_field_closed_probability.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_map_magnetic_field_open_probability() -> np.ndarray:
    """Load the map denoting the probability of an open magnetic field line.

    Returns
    -------
    np.ndarray
        Array of the image.

    Notes
    -----
    This map comes from `MGS data <https://doi.org/10.1029/2007JA012435>`_. It
    has a shape of (180, 360).

    * The zeroth axis corresponds to latitude and spans -90 to 90 degrees.
    * The first axis corresponds to east longitude and spans 0 to 360 degrees.

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import matplotlib.ticker as ticker
       import pyuvs as pu

       fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

       b_field = pu.load_map_magnetic_field_open_probability()
       ax.imshow(b_field, cmap='Blues_r', extent=[0, 360, -90, 90],
                          origin='lower', rasterized=True)
       ax.set_xlabel('Longitude [degrees]')
       ax.set_ylabel('Latitude [degrees]')
       ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
       ax.yaxis.set_major_locator(ticker.MultipleLocator(30))

       plt.show()

    """
    try:
        return np.load(_get_maps_directory() / 'magnetic_field_open_probability.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_map_mars_surface() -> np.ndarray:
    """Load the Mars surface map.

    Returns
    -------
    np.ndarray
        Array of the image.

    Notes
    -----
    The shape of this array is (1800, 3600, 4).

    * The zeroth axis corresponds to latitude and spans -90 to 90 degrees.
    * The first axis corresponds to east longitude and spans 0 to 360 degrees.
    * The second axis is the RGBA channel.

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import matplotlib.ticker as ticker
       import pyuvs as pu

       fig, ax = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)

       mars = pu.load_map_mars_surface()
       ax.imshow(mars, extent=[0, 360, -90, 90], origin='lower', rasterized=True)
       ax.set_xlabel('Longitude [degrees]')
       ax.set_ylabel('Latitude [degrees]')
       ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
       ax.yaxis.set_major_locator(ticker.MultipleLocator(30))

       plt.show()

    """
    try:
        return np.load(_get_maps_directory() / 'mars_surface.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_wavelengths() -> np.ndarray:
    """Load the wavelength centers that correspond to the MUV templates.

    Returns
    -------
    np.ndarray
        Array of the wavelengths.

    Notes
    -----
    The shape of this array is (1024,).

    """
    return np.linspace(174.00487653, 341.44029638, num=1024)


def load_template_co_cameron() -> np.ndarray:
    """Load the normalized MUV CO Cameron bands template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_co_cameron()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'co_cameron_bands.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_co_plus_1st_negative() -> np.ndarray:
    """Load the normalized MUV CO :sup:`+` 1NG (first negative) bands template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_co_plus_1st_negative()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'co+_first_negative.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_co2_plus_fdb() -> np.ndarray:
    """Load the MUV CO :sub:`2` :sup:`+` FDB (Fox-Duffendack-Barker) bands template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_co2_plus_fdb()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'co2+_fox_duffendack_barker.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_co2_plus_uvd() -> np.ndarray:
    """Load the MUV CO :sub:`2` :sup:`+` UVD (ultraviolet doublet) template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_co2_plus_uvd()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'co2+_ultraviolet_doublet.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_n2_vk() -> np.ndarray:
    """Load the MUV N :sub:`2` VK (Vegard-Kaplan) bands template.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_n2_vk()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'nitrogen_vegard_kaplan.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_no_nightglow() -> np.ndarray:
    """Load the MUV NO nightglow bands template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_no_nightglow()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'no_nightglow.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_oxygen_2972() -> np.ndarray:
    """Load the MUV oxygen 297.2 nm template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,).

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_oxygen_2972()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'oxygen_2972.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_template_solar_continuum() -> np.ndarray:
    """Load the MUV solar continuum template.

    This template is in uncalibrated DNs.

    Returns
    -------
    np.ndarray
        Array of the template.

    Notes
    -----
    The shape of this array is (1024,). This is a generic solar continuum and
    it is thus probably best for use in a quicklook. It may be better to use
    a targeted solar continuum for more rigorous data investigation.

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       template = pu.load_template_solar_continuum()
       wavelengths = pu.load_template_wavelengths()
       ax.plot(wavelengths, template)
       ax.set_xlim(wavelengths[0], wavelengths[-1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Relative brightness')

       plt.show()

    """
    try:
        return np.load(_get_templates_directory() / 'solar_continuum.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_voltage_correction_voltage() -> np.ndarray:
    """Load the voltages used in the voltage correction.

    Returns
    -------
    np.ndarray
        Array of the voltages.

    """
    try:
        return np.load(_get_instrument_directory() / 'voltage.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_voltage_correction_coefficients() -> np.ndarray:
    """Load the voltages fit coefficients used in the voltage correction.

    Returns
    -------
    np.ndarray
        Array of the voltage fit coefficients.

    """
    try:
        return np.load(_get_instrument_directory() / 'voltage_fit_coefficients.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_fuv_sensitivity_curve_manufacturer() -> np.ndarray:
    """Load the manufacturer's FUV detector factory sensitivity curve.

    Returns
    -------
    np.ndarray
        Array of the FUV sensitivity curve.

    Notes
    -----
    This array has a shape of (2, 101). Index 0 of the first axis is the
    wavelength corresponding to the sensitivity curve; index 1 is the
    sensitivity curve.

    This is the detector sensitivity in DN / (photons / cm :sup:`2`) at
    gain = 1. The manufacturer reported this curve June 9, 2014.

    See Also
    --------
    load_muv_sensitivity_curve_manufacturer: The MUV analogue to this curve.

    Examples
    --------
    Plot the sensitivity curve.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       curve = pu.load_fuv_sensitivity_curve_manufacturer()
       ax.plot(curve[0], curve[1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Detector Sensitivity')
       ax.set_xlim(100, 200)
       ax.set_ylim(0, 0.1)

       plt.show()

    """
    try:
        return np.load(_get_instrument_directory() / 'fuv_sensitivity_curve_manufacturer.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_muv_flatfield() -> np.ndarray:
    """Load the mid-hi-resolution flatfield created from data taken during the
    Mars year 34 global dust storm.

    Returns
    -------
    np.ndarray
        Array of the flatfield.

    Notes
    -----
    This array has a shape of (133, 19). This flatfield was made from orbits
    7509 to 7523.

    Examples
    --------
    Visualize this flatfield.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots(1, 1, figsize=(8, 2), constrained_layout=True)

       flatfield = pu.load_muv_flatfield()
       ax.pcolormesh(flatfield.T, vmin=0.9, vmax=1.1)
       ax.set_xlabel('Spatial bin')
       ax.set_ylabel('Spectral bin')

       plt.show()

    """
    try:
        return np.load(_get_instrument_directory() / 'muv_flatfield_133x19.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_muv_point_spread_function() -> np.ndarray:
    """Load the MUV point spread function.

    Returns
    -------
    np.ndarray
        Array of the point spread function.

    Notes
    -----
    This array has a shape of (181,). The shape of this array is best
    characterized by a Voigt profile.

    Examples
    --------
    Visualize this array.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import numpy as np
       import pyuvs as pu

       fig, ax = plt.subplots()

       psf = pu.load_muv_point_spread_function()
       detector_pixels = np.arange(181)
       ax.plot(detector_pixels, psf)
       ax.set_xlabel('Detector pixels')
       ax.set_ylabel('Point spread function')

       plt.show()

    """
    try:
        return np.load(_get_instrument_directory() / 'muv_point_spread_function.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_muv_sensitivity_curve_manufacturer() -> np.ndarray:
    """Load the manufacturer's MUV detector factory sensitivity curve.

    Returns
    -------
    np.ndarray
        Array of the MUV sensitivity curve.

    Notes
    -----
    This array has a shape of (2, 101). Index 0 of the first axis is the
    wavelength corresponding to the sensitivity curve; index 1 is the
    sensitivity curve.

    This is the detector sensitivity in DN / (photons / cm :sup:`2`) at
    gain = 1. The manufacturer reported this curve June 9, 2014.

    See Also
    --------
    load_fuv_sensitivity_curve_manufacturer: The FUV analogue to this curve.
    load_muv_sensitivity_curve_observational: This curve created from
                                              observational data.

    Examples
    --------
    Plot the sensitivity curve.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       curve = pu.load_muv_sensitivity_curve_manufacturer()
       ax.plot(curve[0], curve[1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Detector Sensitivity')
       ax.set_xlim(170, 360)
       ax.set_ylim(0, 0.06)

       plt.show()

    """
    try:
        return np.load(_get_instrument_directory() / 'muv_sensitivity_curve_manufacturer.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe


def load_muv_sensitivity_curve_observational() -> np.ndarray:
    """Load the MUV detector sensitivity curve derived from observations.

    Returns
    -------
    np.ndarray
        Array of the MUV sensitivity curve.

    Notes
    -----
    This array has a shape of (2, 512). Index 0 of the first axis is the
    wavelength corresponding to the sensitivity curve; index 1 is the
    sensitivity curve.

    This is the detector sensitivity in DN / (photons / cm :sup:`2`) at
    gain = 1. Justin Deighan reported this curve October 19, 2018.

    See Also
    --------
    load_muv_sensitivity_curve_manufacturer: This curve reported by the
                                             manufacturer.

    Examples
    --------
    Plot the sensitivity curve.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       import pyuvs as pu

       fig, ax = plt.subplots()

       curve = pu.load_muv_sensitivity_curve_observational()
       ax.plot(curve[0], curve[1])
       ax.set_xlabel('Wavelength [nm]')
       ax.set_ylabel('Detector Sensitivity')
       ax.set_xlim(170, 360)
       ax.set_ylim(0, 0.06)

       plt.show()

    """
    try:
        return np.load(_get_instrument_directory() / 'muv_sensitivity_curve_observational.npy')
    except FileNotFoundError as fe:
        raise FileNotFoundError(_error_message) from fe
