import logging
import pytest
import numpy as np
from sklearn.metrics import r2_score
import openturns as ot
from otsurrogate.misc import multi_eval


class Ishigami:
    r"""Ishigami class [Ishigami1990]_.

    .. math:: F = \sin(x_1)+7\sin(x_2)^2+0.1x_3^4\sin(x_1), x\in [-\pi, \pi]^3

    It exhibits strong nonlinearity and nonmonotonicity.
    Depending on `a` and `b`, emphasize the non-linearities.
    It also has a dependence on X3 due to second order interactions (F13).

    """

    logger = logging.getLogger(__name__)

    def __init__(self, a=7., b=0.1):
        """Set up Ishigami.

        :param float a, b: Ishigami parameters
        """
        self.d_in = 3
        self.d_out = 1
        self.a = a
        self.b = b

        var = 0.5 + self.a ** 2 / 8 + self.b * np.pi ** 4 / 5\
            + self.b ** 2 * np.pi ** 8 / 18
        v1 = 0.5 + self.b * np.pi ** 4 / 5 + self.b ** 2 * np.pi ** 8 / 50
        v2 = a ** 2 / 8
        v3 = 0
        v12 = 0
        v13 = self.b ** 2 * np.pi ** 8 * 8 / 225
        v23 = 0

        self.s_first = np.array([v1 / var, v2 / var, v3 / var])
        self.s_second = np.array([[0., 0., v13 / var],
                                  [v12 / var, 0., v23 / var],
                                  [v13 / var, v23 / var, 0.]])
        self.s_total2 = self.s_first + self.s_second.sum(axis=1)
        self.s_total = np.array([0.558, 0.442, 0.244])
        self.logger.info("Using function Ishigami with a={}, b={}"
                         .format(self.a, self.b))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = np.sin(x[0]) + self.a * np.sin(x[1])**2 + \
            self.b * (x[2]**4) * np.sin(x[0])
        return f


class G_Function:
    r"""G_Function class [Saltelli2000]_.

    .. math:: F = \Pi_{i=1}^d \frac{\lvert 4x_i - 2\rvert + a_i}{1 + a_i}

    Depending on the coefficient :math:`a_i`, their is an impact on the impact
    on the output. The more the coefficient is for a parameter, the less the
    parameter is important.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=4, a=None):
        """G-function definition.

        :param int d: input dimension
        :param np.array a: (1, d)
        """
        self.d_in = d
        self.d_out = 1

        if a is None:
            self.a = np.arange(1, d + 1)
        else:
            self.a = np.array(a)

        vi = 1. / (3 * (1 + self.a)**2)
        v = -1 + np.prod(1 + vi)
        self.s_first = vi / v
        self.s_second = np.zeros((self.d_in, self.d_in))
        self.s_total = vi * np.prod(1 + vi) / v

        self.logger.info("Using function G-Function with d={}, a={}"
                         .format(self.d_in, self.a))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 1.
        for i in range(self.d_in):
            f *= (abs(4. * x[i] - 2) + self.a[i]) / (1. + self.a[i])
        return f


class Forrester:
    r"""Forrester class [Forrester2007]_.

    .. math:: F_{e}(x) = (6x-2)^2\sin(12x-4), \\
              F_{c}(x) = AF_e(x)+B(x-0.5)+C,

    were :math:`x\in{0,1}` and :math:`A=0.5, B=10, C=-5`.

    This set of two functions are used to represents a high an a low fidelity.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, fidelity='e'):
        """Forrester-function definition.

        ``e`` stands for expansive and ``c`` for cheap.

        :param str fidelity: select the fidelity ``['e'|'f']``
        """
        self.d_in = 1
        self.d_out = 1
        self.fidelity = fidelity

        self.logger.info('Using function Forrester with fidelity: {}'
                         .format(self.fidelity))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        x = x[0]
        f_e = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        if self.fidelity == 'e':
            return f_e
        else:
            f = 0.5 * f_e + 10 * (x - 0.5) - 5

        return f


class Channel_Flow:
    r"""Channel Flow class.

    .. math:: \frac{dh}{ds}=\mathcal{F}(h)=I\frac{1-(h/h_n)^{-10/3}}{1-(h/h_c)^{-3}}\\
        h_c=\left(\frac{q^2}{g}\right)^{1/3}, h_n=\left(\frac{q^2}{IK_s^2}\right)^{3/10}
    """

    logger = logging.getLogger(__name__)

    def __init__(self, dx=8000., length=40000., width=500., slope=5e-4, hinit=10.):
        """Initialize the geometrical configuration.

        :param float dx: discretization.
        :param float length: Canal length.
        :param float width: Canal width.
        :param float slope: Canal slope.
        :param float hinit: Downstream boundary condition.
        """
        self.w = width
        self.slope = slope
        self.g = 9.8
        self.dx = dx
        self.length = length
        self.x = np.arange(self.dx, self.length + 1, self.dx)
        self.d_out = len(self.x)
        self.d_in = 2
        self.dl = int(self.length // self.dx)
        self.hinit = hinit
        self.zref = - self.x * self.slope

        # Sensitivity
        self.s_first = np.array([0.92925829, 0.05243018])
        self.s_second = np.array([[0., 0.01405351], [0.01405351, 0.]])
        self.s_total = np.array([0.93746788, 0.05887997])

        self.logger.info("Using function Channel Flow with: dx={}, length={}, "
                         "width={}".format(dx, length, width))

    @multi_eval
    def __call__(self, x, h_nc=False):
        """Call function.

        :param list x: inputs [Ks, Q].
        :param bool h_nc: Whether to return hc and hn.
        :return: Water height along the channel.
        :rtype: array_like (n_samples, n_features [+ 2])
        """
        ks, q = x
        hc = np.power((q ** 2) / (self.g * self.w ** 2), 1. / 3.)
        hn = np.power((q ** 2) / (self.slope * self.w ** 2 * ks ** 2), 3. / 10.)

        h = self.hinit * np.ones(self.dl)
        for i in range(2, self.dl + 1):
            h[self.dl - i] = h[self.dl - i + 1] - self.dx * self.slope\
                * ((1 - np.power(h[self.dl - i + 1] / hn, -10. / 3.))
                   / (1 - np.power(h[self.dl - i + 1] / hc, -3.)))

        z_h = self.zref + h

        return np.append(z_h, np.array([hc, hn])) if h_nc else z_h


class Datatest:
    """Wrap results."""

    def __init__(self, kwds):
        self.__dict__.update(kwds)


@pytest.fixture()
def seed():
    np.random.seed(123456)
    ot.RandomGenerator.SetSeed(123456)


@pytest.fixture(scope="module")
def tmp(tmpdir_factory):
    """Create a common temp directory."""
    return str(tmpdir_factory.mktemp('tmp_test'))


@pytest.fixture(scope='session')
def settings_ishigami():
    return {
        "space": {
            "corners": [
                [-np.pi, -np.pi, -np.pi],
                [np.pi, np.pi, np.pi]
            ],
            "sampling": {
                "init_size": 150,
                "method": "halton"
            },
            "resampling": {
                "delta_space": 0.08,
                "resamp_size": 1,
                "method": "sigma",
                "q2_criteria": 0.9
            }
        },
        "pod": {
            "dim_max": 100,
            "tolerance": 0.99,
            "server": None,
            "type": "static"
        },
        "snapshot": {
            "max_workers": 10,
            "plabels": ["x1", "x2", "x3"],
            "flabels": ["F"],
            "provider": {
                "type": "function",
                "module": "batman.tests.plugins",
                "function": "f_ishigami"
            },
            "io": {
                "space_fname": "sample-space.json",
                "space_format": "json",
                "data_fname": "sample-data.json",
                "data_format": "json",
            }
        },
        "surrogate": {
            "predictions": [[0, 2, 1]],
            "method": "kriging"
        },
        "uq": {
            "sample": 2000,
            "test": "Ishigami",
            "pdf": ["Uniform(-3.1415, 3.1415)",
                    "Uniform(-3.1415, 3.1415)",
                    "Uniform(-3.1415, 3.1415)"],
            "type": "aggregated",
            "method": "sobol"
        }
    }


def sampling(bounds, n_samples, dists=None):
    """Sample hypercube with halton."""
    bounds = np.asarray(bounds)
    if dists is None:
        dists = [ot.Uniform(float(bounds[0][i]), float(bounds[1][i]))
                 for i in range(bounds.shape[1])]
    dists = ot.ComposedDistribution(dists)

    sequence_type = ot.LowDiscrepancyExperiment(ot.HaltonSequence(),
                                                dists, n_samples)

    return np.array(sequence_type.generate())


@pytest.fixture(scope='session')
def ishigami_data(settings_ishigami):
    data = {}
    data['func'] = Ishigami()
    x1 = ot.Uniform(-3.1415, 3.1415)
    data['dists'] = [x1] * 3
    data['point'] = [2.20, 1.57, 3]
    data['target_point'] = data['func'](data['point'])
    data['space'] = sampling(settings_ishigami['space']['corners'], 150)
    data['corners'] = settings_ishigami['space']['corners']
    data['plabels'] = settings_ishigami['snapshot']['plabels']
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def g_function_data(settings_ishigami):
    data = {}
    data['func'] = G_Function()
    data['dists'] = [ot.Uniform(0, 1)] * 4
    data['point'] = [0.5, 0.2, 0.7, 0.1]
    data['target_point'] = data['func'](data['point'])
    data['space'] = sampling([[0, 0, 0, 0], [1, 1, 1, 1]], 10)
    data['corners'] = [[0, 0, 0, 0], [1, 1, 1, 1]]
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def mascaret_data(settings_ishigami):
    data = {}
    fun = Channel_Flow()
    data['func'] = lambda x: fun(x).reshape(-1, 5)[:, 0:3]
    data['func'].x = fun.x[0:3]
    x1 = ot.Uniform(15., 60.)
    x2 = ot.Normal(4035., 400.)
    data['dists'] = [x1, x2]
    data['point'] = [31.54, 4237.025]
    data['target_point'] = data['func'](data['point'])[0]
    data['space'] = sampling([[15.0, 2500.0], [60, 6000.0]], 50, data['dists'])
    data['corners'] = [[15.0, 2500.0], [60, 6000.0]]
    data['target_space'] = data['func'](data['space'])
    return Datatest(data)


@pytest.fixture(scope='session')
def mufi_data(settings_ishigami):
    data = {}
    f_e = Forrester('e')
    f_c = Forrester('c')
    data['dists'] = [ot.Uniform(0.0, 1.0)]
    data['point'] = [0.65]
    data['target_point'] = f_e(data['point'])
    data['space'] = np.array([[0, 0.5000], [0, 0.2500], [0, 0.7500], [0, 0.1250],
                              [0, 0.6250], [0, 0.3750], [0, 0.8750], [0, 0.0625],
                              [0, 0.5625], [0, 0.3125], [1, 0.5000], [1, 0.2500],
                              [1, 0.7500], [1, 0.1250], [1, 0.6250], [1, 0.3750],
                              [1, 0.8750], [1, 0.0625], [1, 0.5625], [1, 0.3125],
                              [1, 0.8125], [1, 0.1875], [1, 0.6875], [1, 0.4375],
                              [1, 0.9375]])
    data['corners'] = [[0.0], [1.0]]
    data['plabels'] = ['fidelity', 'x']

    working_space = np.array(data['space'])

    data['target_space'] = np.vstack([f_e(working_space[working_space[:, 0] == 0][:, 1:]),
                                      f_c(working_space[working_space[:, 0] == 1][:, 1:])])
    data['func'] = [f_e, f_c]

    return Datatest(data)


def sklearn_q2(dists, model, surrogate):
    dists = ot.ComposedDistribution(dists)
    experiment = ot.LHSExperiment(dists, 1000)
    sample = np.array(experiment.generate())
    ref = model(sample)
    pred = surrogate(sample)

    return r2_score(ref, pred, multioutput='uniform_average')
