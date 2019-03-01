from .random_variable import RandomVariable, TensorLike
from tensorflow_probability import distributions as tfd
import pymc4 as pm
from typing import List


class Mixture(RandomVariable):
    r"""
    Mixture random variable.

    Often used to model subpopulation heterogeneity

    .. math:: f(x \mid w, \theta) = \sum_{i = 1}^n w_i f_i(x \mid \theta_i)

    ========  ============================================
    Support   :math:`\cap_{i = 1}^n \textrm{support}(f_i)`
    Mean      :math:`\sum_{i = 1}^n w_i \mu_i`
    ========  ============================================

    Parameters
    ----------
    p : array of floats
        p >= 0 and p <= 1
        the mixture weights, in the form of probabilities
    distributions : multidimensional PyMC4 distribution (e.g. `pm.Poisson(...)`)
        or iterable of one-dimensional PyMC4 distributions the
        component distributions :math:`f_1, \ldots, f_n`

    Developer Notes
    ---------------
    Mixture models must implement _base_dist (just as any other RV), but
    we must explicitly return the self._distribution object when implementing
    a new mixture distribution. This ensures that log_prob() calculations work correctly.

    For an example, see below an example taken from the last line of _base_dist in the
    ZeroInflatedPoisson distribution implementation (in discrete.py).

    .. code::

        def _base_dist(self, psi, theta, *args, **kwargs):
            return pm.Mixture(
                p=[psi, 1.0 - psi],
                distributions=[
                    pm.Constant(name="Zero", value=0),
                    pm.Poisson(name="Poisson", mu=theta)
                ],
                name="ZeroInflatedPoisson",
            )._distribution  # <---- this is key!

    Compared to PyMC3's API, the Mixture API is slightly changed to make things
    smoother for end-users.

    Firstly, end-users may find it to be extra work to specify that they want the
    distribution objects for each RV. Hence, the Mixture RV will automatically
    grab out the ._distribution object for each RV object passed in. Hence, users
    need only specify the PyMC4 RV object.

    This first point also makes things hopefully maps better to how end-users abstract
    and think about distributions. Our average user probably doesn't distinguish
    very clearly between an RV and a distribution object, though we know to do so.
    Otherwise, we would not have had questions that Junpeng had to answer on discourse
    regarding how to create mixture distributions in which end-users simply forgot to
    add ``.distribution`` at the end of their distribution calls.

    Secondly, we use the "p" and "distributions", rather than the old "w" and "comp_dists"
    kwargs. During the PyMC4 API development, this is probably the only place where I
    (Eric Ma) have chosen to deviate from the old API, hopefully as an improvement for
    newcomers' mental model of the API.
    """

    def _base_dist(self, p: TensorLike, distributions: List[RandomVariable], *args, **kwargs):
        return tfd.Mixture(
            cat=pm.Categorical(p=p, name="MixtureCategories")._distribution,
            components=[d._distribution for d in distributions],
            name=kwargs.get("name"),
        )
