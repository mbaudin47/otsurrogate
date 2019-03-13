import os
import copy
import pytest
import numpy as np
import numpy.testing as npt
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern, ConstantKernel)
from otsurrogate.surrogate import (PC, Kriging, SklearnRegressor, RBFnet, Evofusion,
                                   SurrogateModel)
from otsurrogate.tests.conftest import (sklearn_q2, sampling)


class Test1d:

    def test_PC_1d(self, ishigami_data):
        sample = sampling(ishigami_data.corners, 2000)
        surrogate = PC(distributions=ishigami_data.dists, sample=sample, degree=10,
                       strategy='LS', stieltjes=False)
        input_ = surrogate.sample
        assert len(input_) == 2000
        output = ishigami_data.func(input_)
        surrogate.fit(input_, output)
        pred = np.array(surrogate.evaluate(ishigami_data.point))
        assert pred == pytest.approx(ishigami_data.target_point, 0.1)

        # Compute predictivity coefficient Q2
        q2 = sklearn_q2(ishigami_data.dists, ishigami_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(1, 0.1)

        surrogate = PC(distributions=ishigami_data.dists, degree=10,
                       strategy='Quad')
        input_ = surrogate.sample
        assert len(input_) == 1331
        output = ishigami_data.func(input_)
        surrogate.fit(input_, output)

        # Compute predictivity coefficient Q2
        q2 = sklearn_q2(ishigami_data.dists, ishigami_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(1, 0.2)

        surrogate = PC(distributions=ishigami_data.dists, degree=10, sample=sample,
                       sparse_param={"max_considered_terms": 130,
                                     "most_significant": 30,
                                     "significance_factor": 5e-5},
                       strategy='SparseLS')
        input_ = surrogate.sample
        assert len(input_) == 2000
        output = ishigami_data.func(input_)
        surrogate.fit(input_, output)

        # Compute predictivity coefficient Q2
        q2 = sklearn_q2(ishigami_data.dists, ishigami_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(1, 0.2)

    def test_GP_1d(self, ishigami_data):
        surrogate = Kriging(ishigami_data.space, ishigami_data.target_space)

        # Test one point evaluation
        pred, _ = np.array(surrogate.evaluate(ishigami_data.point))
        assert pred == pytest.approx(ishigami_data.target_point, 0.2)

        # Compute predictivity coefficient Q2
        def wrap_surrogate(x):
            evaluation, _ = surrogate.evaluate(x)
            return evaluation
        q2 = sklearn_q2(ishigami_data.dists, ishigami_data.func, wrap_surrogate)
        assert q2 == pytest.approx(1, 0.1)

        # Kernel and noise
        surrogate = Kriging(ishigami_data.space, ishigami_data.target_space,
                            kernel=Matern(), noise=0.8)

        surrogate = Kriging(ishigami_data.space, ishigami_data.target_space,
                            kernel=Matern(), noise=True)

        # Optimizer
        surrogate = Kriging(ishigami_data.space, ishigami_data.target_space,
                            global_optimizer=False)
        pred, _ = np.array(surrogate.evaluate(ishigami_data.point))
        assert pred == pytest.approx(ishigami_data.target_point, 0.2)

    def test_sk_regressors_1d(self, ishigami_data):
        # From an object
        l_scale = (1.0,) * 3
        scale_bounds = [(0.01, 100)] * 3
        kernel = ConstantKernel() * Matern(length_scale=l_scale,
                                           length_scale_bounds=scale_bounds)

        regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        surrogate = SklearnRegressor(ishigami_data.space, ishigami_data.target_space,
                                     regressor)

        # Test space evaluation
        pred = np.array(surrogate.evaluate(ishigami_data.space))
        npt.assert_almost_equal(ishigami_data.target_space, pred, decimal=1)

        # From a str
        regressor = 'sklearn.ensemble.RandomForestRegressor()'
        surrogate = SklearnRegressor(ishigami_data.space, ishigami_data.target_space,
                                     regressor)

        q2 = sklearn_q2(ishigami_data.dists, ishigami_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(0.77, 0.2)

    def test_RBFnet_1d(self, ishigami_data):
        surrogate = RBFnet(ishigami_data.space, ishigami_data.target_space)

        # Test one point evaluation
        pred = np.array(surrogate.evaluate(ishigami_data.point))
        assert pred == pytest.approx(ishigami_data.target_point, 0.3)

        # Test space evaluation
        pred = np.array(surrogate.evaluate(ishigami_data.space))
        npt.assert_almost_equal(ishigami_data.target_space, pred, decimal=1)

        # Compute predictivity coefficient Q2
        def wrap_surrogate(x):
            evaluation = surrogate.evaluate(x)
            return evaluation
        q2 = sklearn_q2(ishigami_data.dists, ishigami_data.func, wrap_surrogate)
        assert q2 == pytest.approx(0.86, 0.1)

        surrogate = RBFnet(ishigami_data.space, ishigami_data.target_space, regtree=1)


class TestNd:

    def test_PC_nd(self, mascaret_data):
        sample = sampling(mascaret_data.corners, 1000)
        surrogate = PC(distributions=mascaret_data.dists, sample=sample, degree=10,
                       strategy='LS')
        input_ = surrogate.sample
        output = mascaret_data.func(input_)
        surrogate.fit(input_, output)
        pred = np.array(surrogate.evaluate(mascaret_data.point)).reshape(-1)
        npt.assert_almost_equal(mascaret_data.target_point, pred, decimal=1)

        surrogate = PC(distributions=mascaret_data.dists, degree=5,
                       strategy='Quad')
        input_ = surrogate.sample
        output = mascaret_data.func(input_)
        surrogate.fit(input_, output)

        # Compute predictivity coefficient Q2
        q2 = sklearn_q2(mascaret_data.dists, mascaret_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(1, 0.1)

        surrogate = PC(distributions=mascaret_data.dists, degree=10,
                       sample=sample,
                       strategy='SparseLS')
        input_ = surrogate.sample
        output = mascaret_data.func(input_)
        surrogate.fit(input_, output)

        # Compute predictivity coefficient Q2
        q2 = sklearn_q2(mascaret_data.dists, mascaret_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(1, 0.1)

    def test_GP_nd(self, mascaret_data):
        # Scaling as class SurrogateModel does it
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(np.array(mascaret_data.corners))
        space_scaled = scaler.transform(mascaret_data.space)

        surrogate = Kriging(space_scaled, mascaret_data.target_space,
                            kernel=Matern())

        surrogate = Kriging(space_scaled, mascaret_data.target_space)

        # Compute predictivity coefficient Q2
        def wrap_surrogate(x):
            x_scaled = scaler.transform(x)
            evaluation, _ = surrogate.evaluate(x_scaled)
            return evaluation
        q2 = sklearn_q2(mascaret_data.dists, mascaret_data.func, wrap_surrogate)
        assert q2 == pytest.approx(1, 0.1)

    def test_sk_regressors_nd(self, mascaret_data):
        regressor = 'sklearn.ensemble.RandomForestRegressor()'
        surrogate = SklearnRegressor(mascaret_data.space, mascaret_data.target_space,
                                     regressor)

        q2 = sklearn_q2(mascaret_data.dists, mascaret_data.func, surrogate.evaluate)
        assert q2 == pytest.approx(0.9, 0.1)


def test_SurrogateModel_class(tmp, ishigami_data, settings_ishigami):
    sample = sampling(ishigami_data.corners, 500)
    path = os.path.join(tmp, 'surrogate')
    path_space = os.path.join(tmp, 'space')

    try:
        os.makedirs(path)
    except OSError:
        pass
    try:
        os.makedirs(path_space)
    except OSError:
        pass

    # PC
    pc_settings = {'strategy': 'LS', 'degree': 10,
                   'distributions': ishigami_data.dists, 'sample': sample}
    surrogate = SurrogateModel('pc', ishigami_data.corners,
                               ishigami_data.plabels,
                               **pc_settings)
    input_ = surrogate.predictor.sample
    output = ishigami_data.func(input_)
    surrogate.fit(input_, output)
    pred, sigma = surrogate(ishigami_data.point)
    assert sigma is None
    assert pred == pytest.approx(ishigami_data.target_point, 0.5)
    surrogate.write(path)
    assert os.path.isfile(os.path.join(path, 'surrogate.dat'))

    # Kriging
    surrogate = SurrogateModel('kriging', ishigami_data.corners,
                               ishigami_data.plabels)
    surrogate.fit(ishigami_data.space, ishigami_data.target_space)
    surrogate.write(path)
    assert os.path.isfile(os.path.join(path, 'surrogate.dat'))

    surrogate = SurrogateModel('kriging', ishigami_data.corners,
                               ishigami_data.plabels)
    surrogate.read(path)
    assert surrogate.predictor is not None
    npt.assert_array_equal(surrogate.space, ishigami_data.space)

    pred, _ = surrogate(ishigami_data.point)
    assert pred == pytest.approx(ishigami_data.target_point, 0.2)


def test_quality(mufi_data):
    surrogate = SurrogateModel('rbf', mufi_data.corners,
                               np.array(mufi_data.plabels)[1:])
    surrogate.fit(mufi_data.space[10:, 1:], mufi_data.target_space[10:])

    assert surrogate.estimate_quality()[0] == pytest.approx(1, 0.1)


def test_evofusion(mufi_data):
    f_e, f_c = mufi_data.func
    surrogate = Evofusion(mufi_data.space, mufi_data.target_space)

    # Test one point evaluation
    pred, _ = np.array(surrogate.evaluate(mufi_data.point))
    assert pred == pytest.approx(mufi_data.target_point, 0.1)

    # Compute predictivity coefficient Q2
    def wrap_surrogate(x):
        evaluation, _ = surrogate.evaluate(x)
        return evaluation
    q2 = sklearn_q2(mufi_data.dists, f_e, wrap_surrogate)
    assert q2 == pytest.approx(1, 0.1)
