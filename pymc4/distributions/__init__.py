import os

_backend = 'tensorflow'
if 'PYMC_BACKEND' in os.environ:
    _backend = os.environ['PYMC_BACKEND']

if _backend == 'tensorflow':
    from .tensorflow import *
else:
    raise Exception('Backend %s not supported' % _backend)
