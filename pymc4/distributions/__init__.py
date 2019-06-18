__backend = 'tensorflow'

if __backend == 'base':
    from .base import *
elif __backend == 'tensorflow':
    from .tensorflow import *
else:
    raise Exception('Backend %s not supported' % __backend)
