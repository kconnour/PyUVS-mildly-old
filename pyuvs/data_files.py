import math
from pathlib import Path


class Orbit:
    """A data structure containing info from an orbit number

    Parameters
    ----------
    orbit: int
        The MAVEN orbit number.

    Raises
    ------
    TypeError
        Raised if the input orbit is not an int.

    Examples
    --------
    Create an orbit and get its properties.

    >>> import pyuvs as pu
    >>> orbit = pu.Orbit(3453)
    >>> orbit.code
    'orbit03453'
    >>> orbit.block
    'orbit03400'

    """
    def __init__(self, orbit: int):
        self._orbit = orbit
        self._validate_input()

        self._code = self._make_code()
        self._block = self._make_block()

    def _validate_input(self) -> None:
        if not isinstance(self.orbit, int):
            message = 'The orbit must be an int.'
            raise TypeError(message)

    def _make_code(self) -> str:
        return 'orbit' + f'{self.orbit}'.zfill(5)

    def _make_block(self) -> str:
        block = math.floor(self.orbit / 100) * 100
        return 'orbit' + f'{block}'.zfill(5)

    @property
    def orbit(self) -> int:
        """Get the input orbit.

        """
        return self._orbit

    @property
    def code(self) -> str:
        """Get the IUVS "orbit code" for the input orbit.

        """
        return self._code

    @property
    def block(self) -> str:
        """Get the IUVS orbit block for the input orbit.

        """
        return self._block


def find_latest_apoapse_muv_file_paths_from_block(
        data_directory: Path, orbit: int) -> list[Path]:
    """Find the latest apoapse muv file paths in a given directory, where the
    directory is divided into blocks of data spanning 100 orbits.

    Parameters
    ----------
    data_directory: Path
        The directory where the data blocks are located.
    orbit: int
        The orbit number.

    Returns
    -------
    list[Path]
        The latest apoapse MUV file paths from the given orbit.

    Examples
    --------
    Find the latest files from orbit 3453

    >>> from pathlib import Path
    >>> import pyuvs as pu
    >>> p = Path('/media/kyle/McDataFace/iuvsdata/production')
    >>> f = find_latest_apoapse_muv_file_paths_from_block(p, 3453)
    >>> f[0]
    PosixPath('/media/kyle/McDataFace/iuvsdata/production/orbit03400/mvn_iuv_l1b_apoapse-orbit03453-muv_20160708T044652_v13_r01.fits.gz')

    """
    o = Orbit(orbit)
    return sorted((data_directory / o.block).glob(f'*apoapse*{o.code}*muv*.gz'))
