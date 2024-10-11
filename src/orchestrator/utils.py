import hydra
from omegaconf import DictConfig


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def load_config(config_name: str, config_dir: str):
    """
    Loads specified config from yaml into a python Object.
    """

    with hydra.initialize_config_dir(config_dir, version_base=None):
        dict_config: DictConfig = hydra.compose(config_name)
        config = hydra.utils.instantiate(dict_config, _convert_="all")

    return config
