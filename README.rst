|CI|_ |Codecov|_ |Python|_ |License|_

.. |CI| image:: https://circleci.com/gh/tupui/otsurrogate.svg?style=svg
.. _CI: https://circleci.com/gh/tupui/otsurrogate

.. |Codecov| image:: https://gitlab.com/cerfacs/batman/badges/develop/coverage.svg
.. _Codecov: https://gitlab.com/cerfacs/batman/pipelines

.. |Python| image:: https://img.shields.io/badge/python-2.7,_3.7-blue.svg
.. _Python: https://python.org

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://opensource.org/licenses/MIT

otsurrogate
===========

*otsurrogate* allows to have a common interface for building surrogate models.

.. inclusion-marker-do-not-remove

How to install?
---------------

Using the latest python version is prefered! Then to install::

    git clone git@gitlab.com:cerfacs/otsurrogate.git
    cd otsurrogate
    python setup.py install
    python setup.py test

The testing part is optionnal but is recommanded.

.. note:: If you don't have install priviledge, add ``--user`` option after install.
    But the simplest way might be to use pip or a conda environment.

.. warning:: Depending on your configuration, you might have to export your local path: 
    ``export PATH=$PATH:~/.local/bin``. Care to be taken with both your ``PATH``
    and ``PYTHONPATH`` environment variables. Make sure you do not call different
    installation folders. It is recommanded that you leave your ``PYTHONPATH`` empty.

Dependencies
````````````

The required dependencies are: 

- `Python <https://python.org>`_ >= 2.7 or >= 3.4
- `OpenTURNS <http://www.openturns.org>`_ >= 1.10
- `scikit-learn <http://scikit-learn.org>`_ >= 0.18
- `numpy <http://www.numpy.org>`_ >= 1.13
- `scipy <http://scipy.org>`_ >= 0.15
- `pathos <https://github.com/uqfoundation/pathos>`_ >= 0.2

Testing dependencies are: 

- `pytest <https://docs.pytest.org/en/latest/>`_ >= 2.8
- `mock <https://pypi.python.org/pypi/mock>`_ >= 2.0

Extra testing flavours: 

- `coverage <http://coverage.readthedocs.io>`_ >= 4.4
- `pylint <https://www.pylint.org>`_ >= 1.6.0
