_MODELS = {}

def register_model(cls=None, *, dataset=None, name=None):
    """
    A decorator for registering model classes.
    """
    def _register(cls):
        if dataset in _MODELS and name in _MODELS[dataset]:
            raise ValueError(f'Already registered model with name: {name}')
        if dataset not in _MODELS:
            _MODELS[dataset] = {}
        _MODELS[dataset][name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def get_model(dataset, name):
    return _MODELS[dataset][name]
