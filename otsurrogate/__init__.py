"""
otsurrogate package
*******************
"""
import openturns as ot
from .surrogate import SurrogateModel
from .pod import Pod

__all__ = ['SurrogateModel', 'Pod']
__version__ = '1'

ot.RandomGenerator.SetSeed(123456)
