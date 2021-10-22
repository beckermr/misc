from __future__ import print_function
import numpy as np


class NoDataError(Exception):
    """
    there was no data
    """
    def __init__(self, value):
        super(NoDataError, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front == '':
            front = None
        if back == '':
            back = None

        self.front = front
        self.back = back

        if self.front is None and self.back is None:
            self.nomod = True
        else:
            self.nomod = False

    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)
        return n


def get_masked_frac_sums(obs):
    weight = obs.weight
    wmasked = np.where(weight <= 0.0)
    nmasked = wmasked[0].size
    npix = weight.size

    return npix, nmasked


def get_masked_frac(mbobs):
    nmasked = 0.0
    npix = 0

    for obslist in mbobs:
        for obs in obslist:
            tnpix, tnmasked = get_masked_frac_sums(obs)
            nmasked += tnmasked
            npix += tnpix

    masked_frac = nmasked/npix
    return masked_frac
