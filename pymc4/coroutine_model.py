import biwrap
import functools

from pymc4 import name_scope


@biwrap.biwrap
def model(genfn, *, name=None, keep_auxiliary=True, keep_return=True, method=False):
    if method:
        template = ModelTemplate(genfn, name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return)

        @functools.wraps(genfn)
        def wrapped(*args, **kwargs):
            return template(*args, **kwargs)
        return wrapped
    else:
        template = ModelTemplate(genfn, name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return)
        return template


def get_name(default, base_fn, name):
    if name is None:
        if default is not None:
            name = default
        elif hasattr(base_fn, "name"):
            name = getattr(base_fn, "name")
        elif hasattr(base_fn, "__name__"):
            name = base_fn.__name__
    return name


class ModelTemplate(object):
    """
    The `ModelTemplate` object, this keeps a function, responsible for a specific generative process.
    Generative process is one that samples from prior distributions and interacts
    them in a user defined way arbitrarily complex.

    Parameters
    ----------
    genfn : callable
        generative process, that accepts any arguments as conditioners and returns realizations if any.
    keep_auxiliary : bool
        generative process may require some auxiliary variables to be created, but they are probably will not be used
        anywhere else. In that case it is useful to tell PyMC4 engine that we can get rid of auxiliary variables
        as long as they are not needed any more.
    """

    def __init__(self, template, *, name=None, keep_auxiliary=True, keep_return=True):
        self.template = template
        self.name = name
        self.keep_auxiliary = keep_auxiliary
        self.keep_return = keep_return

    def __call__(self, *args, name=None, keep_auxiliary=None, keep_return=None, **kwargs):
        """
        Parameters
        ----------
        name : str
            The desired name for the model, by default, it is inferred from the model declaration context,
            but can be used just once
        keep_auxiliary : bool
            Whether to override the default variable for `keep_auxiliary`
        args : tuple
            positional conditioners for generative process
        kwargs : dict
            keyword conditioners for the generative process

        Returns
        -------
        Model
            The conditioned generative process, for which we can obtain generator (generative process) with :code:`iter`
        """
        genfn = functools.partial(self.template, *args, **kwargs)
        name = get_name(self.name, self.template, name)
        # throw an informative message
        if keep_auxiliary is None:
            keep_auxiliary = self.keep_auxiliary
        if keep_return is None:
            keep_return = keep_return
        return Model(
            genfn, name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return,
        )


class Model(object):
    """
    A container for generators, and the corresponding model
    """
    def __init__(self, genfn, *, name=None, keep_auxiliary=True, keep_return=True):
        self.genfn = genfn
        self.name = name
        self.keep_auxiliary = keep_auxiliary
        self.keep_return = keep_return

    def __iter__(self):
        with name_scope(self.name):
            # correctly handles the case when gen is just a single Distribution object
            # in that case we should immediately proceed sampling and other stuff
            control_flow = self.genfn()
            ret = yield from control_flow
            return ret

    # This will result into unnamed (outer) name scope and not tracked return value
    def __add__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value + other

    def __radd__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other + value

    def __sub__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value - other

    def __rsub__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other - value

    def __mul__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value * other

    def __rmul__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other * value

    def __matmul__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value @ other

    def __rmatmul__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other @ value

    def __truediv__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value / other

    def __rtruediv__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other / value

    def __floordiv__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value // other

    def __rfloordiv__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other // value

    def __mod__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value % other

    def __rmod__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other % value

    def __pow__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value ** other

    def __rpow__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other ** value

    def __and__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value & other

    def __rand__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other & value

    def __xor__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value ^ other

    def __rxor__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other ^ value

    def __or__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return value | other

    def __ror__(self, other):
        value = yield self
        if isinstance(other, Model):
            other = yield other
        return other | value

    def __neg__(self):
        value = yield self
        return -value

    def __pos__(self):
        value = yield self
        return +value

    def __invert__(self):
        value = yield self
        return ~value

    def __getitem__(self, slice_spec, var=None):
        value = yield self
        return value.__getitem__(slice_spec, var=var)
