"""
Misc module
***********
"""
import os
import sys
import time
import warnings
import inspect
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from pathos.multiprocessing import cpu_count
from .nested_pool import NestedPool

__all__ = ['NestedPool', 'cpu_system', 'save_show', 'multi_eval']


def cpu_system():
    """Determine the number of CPU of system."""
    try:
        n_cpu_system = cpu_count()
    except NotImplementedError:
        n_cpu_system = os.sysconf('SC_NPROCESSORS_ONLN')
    return 1 if n_cpu_system == 0 or n_cpu_system is None else n_cpu_system


class ProgressBar:
    """Print progress bar in console."""

    def __init__(self, total):
        """Create a bar.

        :param int total: number of iterations
        """
        self.total = total
        self.calls = 1
        self.progress = 0.

        sys.stdout.write("Progress | " +
                         " " * 50 +
                         " |" + "0.0% ")

        self.init_time = time.time()

    def __call__(self):
        """Update bar."""
        self.progress = (self.calls) / float(self.total) * 100

        eta, vel = self.compute_eta()
        self.show_progress(eta, vel)

        self.calls += 1

    def compute_eta(self):
        """Compute ETA.

        Compare current time with init_time.

        :return: eta, vel
        :rtype: str
        """
        end_time = time.time()
        iter_time = (end_time - self.init_time) / self.calls

        eta = (self.total - self.calls) * iter_time
        eta = time.strftime("%H:%M:%S", time.gmtime(eta))

        vel = str(1. / iter_time)

        return eta, vel

    def show_progress(self, eta=None, vel=None):
        """Print bar and ETA if relevant.

        :param str eta: ETA in H:M:S
        :param str vel: iteration/second
        """
        p_bar = int(np.floor(self.progress / 2))
        sys.stdout.write("\rProgress | " +
                         "-" * (p_bar - 1) + "~" +
                         " " * (50 - p_bar - 1) +
                         " |" + str(self.progress) + "% ")

        if self.progress == 100:
            sys.stdout.write('\n')
            del self
        elif eta and vel:
            sys.stdout.write("| ETA: " + eta + " (at " + vel + " it/s) ")

        sys.stdout.flush()


def save_show(fname, figures, **kwargs):
    """Either show or save the figure[s].

    If :attr:`fname` is `None` the figure will show.

    :param str fname: whether to export to filename or display the figures.
    :param list(Matplotlib figure instance) figures: Figures to handle.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fig in figures:
            try:
                fig.tight_layout()
            except ValueError:
                pass

    if fname is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
        for fig in figures:
            pdf.savefig(fig, transparent=True, bbox_inches='tight', **kwargs)
        pdf.close()
    else:
        plt.show()
    plt.close('all')


def multi_eval(fun):
    """Detect space or unique point.

    Return the evaluation with shape (n_samples, n_features).
    """
    def wrapper_fun(self, x_n, *args, **kwargs):
        """Get evaluation from space or point.

        If the function is a Kriging instance, get and returns the variance.

        :return: function evaluation(s) [sigma(s)]
        :rtype: np.array([n_eval], n_feature)
        """
        x_n = np.atleast_2d(x_n)
        shape_eval = (len(x_n), -1)

        full = kwargs['full'] if 'full' in kwargs else False

        feval = [fun(self, x_i, *args, **kwargs) for x_i in x_n]

        if any(method in inspect.getmodule(fun).__name__
               for method in ['kriging', 'multifidelity']) or full:
            feval, sigma = zip(*feval)
            feval = np.array(feval).reshape(shape_eval)
            sigma = np.array(sigma).reshape(shape_eval)
            return feval, sigma
        feval = np.array(feval).reshape(shape_eval)
        return feval
    return wrapper_fun
