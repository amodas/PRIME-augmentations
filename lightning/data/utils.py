_DATASETS = {}
_CC_DATASETS = {}

def register_dataset(cls=None, *, name=None):
    """
    A decorator for registering dataset classes.
    """
    def _register(cls):
        if name in _DATASETS:
            raise ValueError(f'Already registered dataset with name: {name}')
        _DATASETS[name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def register_cc_dataset(cls=None, *, name=None):
    """
    A decorator for registering common corruptions dataset classes.
    """

    def _register(cls):
        if name in _CC_DATASETS:
            raise ValueError(f'Already registered CC dataset with name: {name}')
        _CC_DATASETS[name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def get_dataset(name):
    return _DATASETS[name]

def get_cc_dataset(name):
    return _CC_DATASETS[name]