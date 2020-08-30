"""Test suite for GP Module"""

import numpy as np
import numpy.testing as npt
import tensorflow as tf
import pymc4 as pm
from pymc4.gp.cov import ExpQuad
from pymc4.gp import LatentGP, MarginalGP


class TestLatentGP:
    def setup(self):
        self.X = np.linspace(0, 1, 10)[:, np.newaxis]
        self.Xuni = np.linspace(0, 1, 1)[:, np.newaxis]
        X1 = np.linspace(0, 1, 10)[:, np.newaxis]
        X2 = np.linspace(0, 5, 10)[:, np.newaxis]
        X3 = np.linspace(0, 10, 10)[:, np.newaxis]
        self.Xmv = np.stack([X1, X2, X3], axis=0)
        self.Xnew = np.linspace(0, 1, 20)[:, np.newaxis]
        self.Xnewuni = np.linspace(0, 1, 1)[:, np.newaxis]
        Xnew1 = np.linspace(0, 1, 20)[:, np.newaxis]
        Xnew2 = np.linspace(0, 5, 20)[:, np.newaxis]
        Xnew3 = np.linspace(0, 10, 20)[:, np.newaxis]
        self.Xnewmv = np.stack([Xnew1, Xnew2, Xnew3], axis=0)
        self.ls = np.array(1.0)
        self.k = ExpQuad(length_scale=self.ls)
        self.gp = LatentGP(cov_fn=self.k)

        @pm.model
        def model():
            ls = yield pm.HalfCauchy("ls", scale=5.0)
            k = ExpQuad(length_scale=ls)
            gp = LatentGP(cov_fn=k)
            f = yield gp.prior("f", self.X.astype(np.float32))
            fcond = yield gp.conditional(
                "fcond",
                self.Xnew.astype(np.float32),
                given={"X": self.X.astype(np.float32), "f": f},
            )

        self.model = model

    def test_prior(self, tf_seed):
        f = self.gp.prior("f", self.X, jitter=1e-8)
        samples = f.sample(5)
        assert isinstance(f, pm.MvNormalCholesky)
        assert f.batch_shape == ()
        assert f.event_shape == (10,)
        assert samples.shape == (5, 10)
        assert not np.isnan(samples).any()

    def test_prior_univariate(self, tf_seed):
        f = self.gp.prior("f", self.Xuni, jitter=1e-8)
        samples = f.sample(5)
        assert isinstance(f, pm.Normal)
        assert f.batch_shape == ()
        assert f.event_shape == ()
        assert samples.shape == (5,)
        assert not np.isnan(samples).any()

    def test_prior_batched(self, tf_seed):
        f = self.gp.prior("f", self.Xmv, jitter=1e-8)
        samples = f.sample(5)
        assert isinstance(f, pm.MvNormalCholesky)
        assert f.batch_shape == (3,)
        assert f.event_shape == (10,)
        assert samples.shape == (5, 3, 10)
        assert not np.isnan(samples).any()

    def test_conditional(self, tf_seed):
        f = self.gp.prior("f", self.X, jitter=1e-8)
        f_sample = f.sample()
        fcond = self.gp.conditional("fcond", self.Xnew, given={"X": self.X, "f": f_sample})
        samples = fcond.sample(5)
        assert isinstance(fcond, pm.MvNormalCholesky)
        assert fcond.batch_shape == ()
        assert fcond.event_shape == (20,)
        assert samples.shape == (5, 20)
        assert not np.isnan(samples).any()

    def test_conditional_univariate(self, tf_seed):
        f = self.gp.prior("f", self.X, jitter=1e-8)
        f_sample = f.sample()
        fcond = self.gp.conditional("fcond", self.Xnewuni, given={"X": self.X, "f": f_sample})
        samples = fcond.sample(5)
        assert isinstance(fcond, pm.Normal)
        assert fcond.batch_shape == ()
        assert fcond.event_shape == ()
        assert samples.shape == (5,)
        assert not np.isnan(samples).any()

    def test_conditional_batched(self, tf_seed):
        f = self.gp.prior("f", self.X, jitter=1e-8)
        f_sample = f.sample()
        fcond = self.gp.conditional("fcond", self.Xnewmv, given={"X": self.X, "f": f_sample})
        samples = fcond.sample(5)
        assert isinstance(fcond, pm.MvNormalCholesky)
        assert fcond.batch_shape == (3,)
        assert fcond.event_shape == (20,)
        assert samples.shape == (5, 3, 20)
        assert not np.isnan(samples).any()

    def test_sampling(self, tf_seed):
        m = self.model()
        trace = pm.sample(m, num_samples=10, num_chains=1, burn_in=10)
        f_samples = np.asarray(trace.posterior["model/f"])
        fcond_samples = np.asarray(trace.posterior["model/fcond"])
        assert f_samples is not None
        assert f_samples.shape == (1, 10, 10)
        assert not np.isnan(f_samples).any()
        assert fcond_samples is not None
        assert fcond_samples.shape == (1, 10, 20)
        assert not np.isnan(fcond_samples).any()


class TestMarginalGP:
    def setup(self):
        self.X = np.linspace(0, 1, 10)[:, np.newaxis]
        self.y = np.random.rand(self.X.shape[0])
        self.Xuni = np.linspace(0, 1, 1)[:, np.newaxis]
        self.yuni = np.random.rand(self.Xuni.shape[0])
        X1 = np.linspace(0, 1, 10)[:, np.newaxis]
        X2 = np.linspace(0, 5, 10)[:, np.newaxis]
        X3 = np.linspace(0, 10, 10)[:, np.newaxis]
        self.Xmv = np.stack([X1, X2, X3], axis=0)
        self.ymv = np.random.rand(3, 10)
        self.Xnew = np.linspace(0, 1, 20)[:, np.newaxis]
        self.Xnewuni = np.linspace(0, 1, 1)[:, np.newaxis]
        Xnew1 = np.linspace(0, 1, 20)[:, np.newaxis]
        Xnew2 = np.linspace(0, 5, 20)[:, np.newaxis]
        Xnew3 = np.linspace(0, 10, 20)[:, np.newaxis]
        self.Xnewmv = np.stack([Xnew1, Xnew2, Xnew3], axis=0)
        self.ls = np.array(1.0)
        self.noise = np.array(1e-6)
        self.k = ExpQuad(length_scale=self.ls)
        self.gp = MarginalGP(cov_fn=self.k)

        @pm.model
        def model():
            ls = yield pm.HalfCauchy("ls", scale=5.0)
            k = ExpQuad(length_scale=ls)
            gp = MarginalGP(cov_fn=k)
            sigma = yield pm.Beta("sigma", 1.0, 1.0)
            y_ = yield gp.marginal_likelihood(
                "y_", self.X.astype(np.float32), self.y.astype(np.float32), noise=sigma, jitter=0
            )
            y_pred = yield gp.conditional("y_pred", self.Xnew.astype(np.float32))

        self.model = model

    def test_marginal_likelihood(self, tf_seed):
        y_ = self.gp.marginal_likelihood("y_", self.X, self.y, noise=self.noise, jitter=1e-8)
        samples = y_.sample(5)
        assert isinstance(y_, pm.MvNormalCholesky)
        assert y_.batch_shape == ()
        assert y_.event_shape == (10,)
        assert y_.is_observed == True
        assert samples.shape == (5, 10)
        assert not np.isnan(samples).any()

    def test_marginal_likelihood_univariate(self, tf_seed):
        y_ = self.gp.marginal_likelihood("y_", self.Xuni, self.yuni, noise=self.noise, jitter=1e-8)
        samples = y_.sample(5)
        assert isinstance(y_, pm.Normal)
        assert y_.batch_shape == ()
        assert y_.event_shape == ()
        assert y_.is_observed == True
        assert samples.shape == (5,)
        assert not np.isnan(samples).any()

    def test_prior_batched(self, tf_seed):
        y_ = self.gp.marginal_likelihood("y_", self.Xmv, self.ymv, noise=self.noise, jitter=1e-8)
        samples = y_.sample(5)
        assert isinstance(y_, pm.MvNormalCholesky)
        assert y_.batch_shape == (3,)
        assert y_.event_shape == (10,)
        assert y_.is_observed == True
        assert samples.shape == (5, 3, 10)
        assert not np.isnan(samples).any()

    def test_conditional(self, tf_seed):
        y_pred = self.gp.conditional(
            "y_pred", self.Xnew, given={"X": self.X, "y": self.y, "noise": self.noise}
        )
        samples = y_pred.sample(5)
        assert isinstance(y_pred, pm.MvNormalCholesky)
        assert y_pred.batch_shape == ()
        assert y_pred.event_shape == (20,)
        assert samples.shape == (5, 20)
        assert not np.isnan(samples).any()

    def test_conditional_univariate(self, tf_seed):
        y_pred = self.gp.conditional(
            "y_pred", self.Xnewuni, given={"X": self.X, "y": self.y, "noise": self.noise}
        )
        samples = y_pred.sample(5)
        assert isinstance(y_pred, pm.Normal)
        assert y_pred.batch_shape == ()
        assert y_pred.event_shape == ()
        assert samples.shape == (5,)
        assert not np.isnan(samples).any()

    def test_conditional_batched(self, tf_seed):
        y_pred = self.gp.conditional(
            "y_pred", self.Xnewmv, given={"X": self.X, "y": self.y, "noise": self.noise}
        )
        samples = y_pred.sample(5)
        assert isinstance(y_pred, pm.MvNormalCholesky)
        assert y_pred.batch_shape == (3,)
        assert y_pred.event_shape == (20,)
        assert samples.shape == (5, 3, 20)
        assert not np.isnan(samples).any()

    def test_sampling(self, tf_seed):
        m = self.model()
        trace = pm.sample(m, num_samples=10, num_chains=1, burn_in=10)
        ppc = pm.sample_posterior_predictive(m, trace, ["model/y_", "model/y_pred"])
        y_samples = np.asarray(ppc.posterior_predictive["model/y_"])
        y_pred_samples = np.asarray(ppc.posterior_predictive["model/y_pred"])
        assert y_samples is not None
        assert y_samples.shape == (1, 10, 10)
        assert not np.isnan(y_samples).any()
        assert y_pred_samples is not None
        assert y_pred_samples.shape == (1, 10, 20)
        assert not np.isnan(y_pred_samples).any()

    def test_predict(self, tf_seed):
        X = np.linspace(0, 1, 10)[:, np.newaxis]
        Xnew = np.linspace(0, 1, 20)[:, np.newaxis]
        y = np.random.rand(X.shape[0])
        ls = np.array(1.0)
        noise = np.array(1e-06)
        k = ExpQuad(length_scale=ls)
        gp = MarginalGP(cov_fn=k)
        samples = gp.predict(Xnew, given={"X": X, "y": y, "noise": noise})
        assert samples.shape == (20,)
        assert not np.isnan(samples).any()

    def test_predictt(self, tf_seed):
        X = np.linspace(0, 1, 10)[:, np.newaxis]
        Xnew = np.linspace(0, 1, 20)[:, np.newaxis]
        y = np.random.rand(X.shape[0])
        ls = np.array(1.0)
        noise = np.array(1e-06)
        k = ExpQuad(length_scale=ls)
        gp = MarginalGP(cov_fn=k)
        mu, cov = gp.predictt(Xnew, given={"X": X, "y": y, "noise": noise})
        assert mu.shape == (20,)
        assert cov.shape == (20, 20)
        assert not np.isnan(cov).any()
        assert not np.isnan(mu).any()
        npt.assert_(np.all(tf.linalg.diag_part(cov) > 0.0))
        npt.assert_allclose(cov.numpy(), cov.numpy().T, rtol=1e-5)
        npt.assert_(np.all(np.linalg.eigvals(cov) > 0.0))

        # test with to_numpy=True
        mu, cov = gp.predictt(Xnew, given={"X": X, "y": y, "noise": noise}, to_numpy=True)
        assert isinstance(mu, np.ndarray)
        assert isinstance(cov, np.ndarray)
