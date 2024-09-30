import hydra
from omegaconf import DictConfig


def load_config(config_name: str, config_dir: str):
    """
    Loads specified config from yaml into a python Object.
    """

    with hydra.initialize_config_dir(config_dir, version_base=None):
        dict_config: DictConfig = hydra.compose(config_name)
        config = hydra.utils.instantiate(dict_config, _convert_="all")

    return config
