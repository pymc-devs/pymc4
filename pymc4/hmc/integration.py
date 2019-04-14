from collections import namedtuple
from scipy import linalg


State = namedtuple("State", "q, p, v, q_grad, energy, model_logp")


class IntegrationError(RuntimeError):
    pass


class CpuLeapfrogIntegrator(object):
    def __init__(self, potential, logp_dlogp_func):
        """Leapfrog integrator using CPU."""
        self._potential = potential
        self._logp_dlogp_func = logp_dlogp_func

    def compute_state(self, q, p):
        """Compute Hamiltonian functions using a position and momentum."""
        logp, dlogp = self._logp_dlogp_func(q)
        v = self._potential.velocity(p)
        kinetic = self._potential.energy(p, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, dlogp, energy, logp)

    def step(self, epsilon, state, out=None):
        """Leapfrog integrator step.

        Half a momentum update, full position update, half momentum update.

        Parameters
        ----------
        epsilon: float, > 0
            step scale
        state: State namedtuple,
            current position data
        out: (optional) State namedtuple,
            preallocated arrays to write to in place

        Returns
        -------
        None if `out` is provided, else a State namedtuple
        """
        try:
            return self._step(epsilon, state)
        except linalg.LinAlgError as err:
            msg = "LinAlgError during leapfrog step."
            raise IntegrationError(msg)
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                msg = "Infs or nans in scipy.linalg during leapfrog step."
                raise IntegrationError(msg)
            else:
                raise

    def _step(self, epsilon, state):
        pot = self._potential
        q, p, v, q_grad, energy, logp = state

        dt = 0.5 * epsilon

        # half momentum step
        p_new = p + dt * q_grad

        # whole position step
        v_new = pot.velocity(p_new)
        q_new = (q + epsilon * v_new).astype(q.dtype)

        # half momentum step
        logp, q_new_grad = self._logp_dlogp_func(q_new)
        p_new = p_new + dt * q_new_grad

        kinetic = pot.velocity_energy(p_new, v_new)
        energy = kinetic - logp

        return State(q_new, p_new, v_new, q_new_grad, energy, logp)
