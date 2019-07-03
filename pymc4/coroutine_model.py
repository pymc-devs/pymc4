import functools
import types
import pymc4
from pymc4.scopes import name_scope
from pymc4.utils import biwrap, NameParts


@biwrap
def model(genfn, *, name=None, keep_auxiliary=True, keep_return=True, method=False):
    if method:
        template = ModelTemplate(
            genfn, name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return
        )

        @functools.wraps(genfn)
        def wrapped(*args, **kwargs):
            return template(*args, **kwargs)

        return wrapped
    else:
        template = ModelTemplate(
            genfn, name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return
        )
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
    Model Template -- generative model with metadata.

    ModelTemplate is a callable object that represents a generative process. A generative process samples
    from prior distributions and allows them to interact in arbitrarily-complex, user-defined ways.

    Parameters
    ----------
    template : callable
        generative process, that accepts any arguments as conditioners and returns realizations if any.
    keep_auxiliary : bool
        generative process may require some auxiliary variables to be created, but they are probably will not be used
        anywhere else. In that case it is useful to tell PyMC4 engine that we can get rid of auxiliary variables
        as long as they are not needed any more.
    keep_return : bool
        the return value of the model will be recorded
    """

    def __init__(self, template, *, name=None, keep_auxiliary=True, keep_return=True):
        self.template = template
        self.name = name
        self.keep_auxiliary = keep_auxiliary
        self.keep_return = keep_return

    def __call__(self, *args, name=None, keep_auxiliary=None, keep_return=None, **kwargs):
        """
        Evaluate the model.

        Model evaluation usually comes with :code:`yield` keyword, see Examples below

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

        Examples
        --------
        >>> import pymc4 as pm
        >>> from pymc4 import distributions as dist

        >>> @pm.model(keep_return=False)  # do not keep `norm` in return
        ... def nested_model(cond):
        ...     norm = yield dist.Normal("n", cond, 1)
        ...     return norm

        >>> @pm.model  # keep_return is True by default
        ... def main_model():
        ...     norm = yield dist.Normal("n", 0, 1)
        ...     result = yield nested_model(norm, name="a")
        ...     return result
        >>> ret, state = pm.evaluate_model(main_model())
        >>> assert "main_model" in state.values
        >>> assert "main_model/a" not in state.values
        """
        genfn = functools.partial(self.template, *args, **kwargs)
        name = get_name(self.name, self.template, name)
        if name is not None and not NameParts.is_valid_untransformed_name(name):
            # throw an informative message to fix a name
            raise ValueError(NameParts.UNTRANSFORMED_NAME_ERROR_MESSAGE)
        if keep_auxiliary is None:
            keep_auxiliary = self.keep_auxiliary
        if keep_return is None:
            keep_return = self.keep_return
        return Model(genfn, name=name, keep_auxiliary=keep_auxiliary, keep_return=keep_return)


def unpack(arg):
    if isinstance(arg, (Model, types.GeneratorType)):
        return (yield arg)
    else:
        return arg


def yieldify(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        args, kwargs = pymc4.utils.map_nested(unpack, (args, kwargs))
        return fn(*args, **kwargs)

    return wrapped


class Model(object):
    # this is gonna be used for generator-like objects
    _default_model_info = dict(keep_auxiliary=True, keep_return=False)

    def __init__(self, genfn, *, name=None, keep_auxiliary=True, keep_return=True):
        self.genfn = genfn
        self.name = name
        self._model_info = dict(keep_auxiliary=keep_auxiliary, keep_return=keep_return)

    def model_info(self):
        info = self._model_info.copy()
        info.update(scope=name_scope(self.name), name=self.name)
        return info

    @classmethod
    def default_model_info(cls):
        info = cls._default_model_info.copy()
        info.update(scope=name_scope(None), name=None)
        return info

    def control_flow(self):
        return (yield from self.genfn())

    def __iter__(self):
        return self.control_flow()

    # This will result into unnamed (outer) name scope and not tracked return value
    @yieldify
    def __add__(self, other):
        return self + other

    @yieldify
    def __radd__(self, other):
        return other + self

    @yieldify
    def __sub__(self, other):
        return self - other

    @yieldify
    def __rsub__(self, other):
        return other - self

    @yieldify
    def __mul__(self, other):
        return self * other

    @yieldify
    def __rmul__(self, other):
        return other * self

    @yieldify
    def __matmul__(self, other):
        return self @ other

    @yieldify
    def __rmatmul__(self, other):
        return other @ self

    @yieldify
    def __truediv__(self, other):
        return self / other

    @yieldify
    def __rtruediv__(self, other):
        return other / self

    @yieldify
    def __floordiv__(self, other):
        return self // other

    @yieldify
    def __rfloordiv__(self, other):
        return other // self

    @yieldify
    def __mod__(self, other):
        return self % other

    @yieldify
    def __rmod__(self, other):
        return other % self

    @yieldify
    def __pow__(self, other):
        return self ** other

    @yieldify
    def __rpow__(self, other):
        return other ** self

    @yieldify
    def __and__(self, other):
        return self & other

    @yieldify
    def __rand__(self, other):
        return other & self

    @yieldify
    def __xor__(self, other):
        return self ^ other

    @yieldify
    def __rxor__(self, other):
        return other ^ self

    @yieldify
    def __or__(self, other):
        return self | other

    @yieldify
    def __ror__(self, other):
        return other | self

    @yieldify
    def __neg__(self):
        return -self

    @yieldify
    def __pos__(self):
        return +self

    @yieldify
    def __invert__(self):
        return ~self

    @yieldify
    def __getitem__(self, slice_spec, var=None):
        return self.__getitem__(slice_spec, var=var)
