from .specs import SPECS
from .sampler import SAMPLERS

ALGORITHMS = list(SAMPLERS.keys())

_DATA_EXPORTS = {
    "SALSACLRSDataModule",
    "SALSACLRSDataset",
    "DynamicDataset",
    "SALSACLRSDataLoader",
    "load_dataset",
}


def __getattr__(name):
    if name in _DATA_EXPORTS:
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
