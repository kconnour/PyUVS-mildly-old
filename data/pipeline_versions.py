from pathlib import Path

import h5py
import yaml


def get_path_of_pipeline_versions() -> Path:
    return Path(__file__).resolve().parent / 'pipeline_versions.yaml'


def get_latest_pipeline_versions() -> dict:
    with open(get_path_of_pipeline_versions()) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def current_file_is_up_to_date(hdf5_file: h5py.File, pipeline: str) -> bool:
    latest_pipeline_versions = get_latest_pipeline_versions()
    return hdf5_file['versions'].attrs[pipeline] == latest_pipeline_versions[pipeline]
