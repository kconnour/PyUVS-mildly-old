from pathlib import Path

from file_structure import make_hdf5_file
from fits_to_hdf5 import fill_hdf5_file_with_fits_info


if __name__ == '__main__':
    # Example usage:
    hdf5_loc = Path('/media/kyle/iuvs/apoapse')
    fits_loc = Path('/media/kyle/iuvs/production')
    for orbit in range(4002, 4003):
        print(orbit)
        make_hdf5_file(orbit, hdf5_loc, fits_loc)
        fill_hdf5_file_with_fits_info(orbit, hdf5_loc, fits_loc)
