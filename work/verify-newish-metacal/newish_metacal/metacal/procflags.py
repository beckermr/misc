from __future__ import print_function
import numpy as np

# no attempt was made at all
NO_ATTEMPT = 2**0

# no data was available, e.g. if there
# were no epochs at all
NO_DATA = 2**1

# No good images were available, e.g. if there was some
# kind of blacklist used
IMAGE_FLAGS = 2**2

# none of the epochs got a psf fit
PSF_FAILURE = 2**3

# the object/fof fitting failed
OBJ_FAILURE = 2**4

# the masked fraction was too high.  This is checked
# *before* additional masking is set, e.g. in _set_weight
HIGH_MASKFRAC = 2**5

# Too few pixels available for fitting
# occurs in the _set_weight method
TOO_FEW_PIXELS = 2**6

# all epochs had the center masked
ALL_CENTERS_MASKED = 2**7

FLAG_MAP = {
    'ok': 0,
    0: 'ok',
    'no_attempt': NO_ATTEMPT,
    NO_ATTEMPT: 'no_attempt',

    'no_data': NO_DATA,
    NO_DATA: 'no_data',

    'image_flags': IMAGE_FLAGS,
    IMAGE_FLAGS: 'image_flags',

    'psf_failure': PSF_FAILURE,
    PSF_FAILURE: 'psf_failure',

    'obj_failure': OBJ_FAILURE,
    OBJ_FAILURE: 'obj_failure',

    'high_maskfrac': HIGH_MASKFRAC,
    HIGH_MASKFRAC: 'high_maskfrac',

    'too_few_pixels': TOO_FEW_PIXELS,
    TOO_FEW_PIXELS: 'too_few_pixels',

    'all_centers_masked': ALL_CENTERS_MASKED,
    ALL_CENTERS_MASKED: 'all_centers_masked',
}

_numorder = [key for key in FLAG_MAP if isinstance(key, int)]
_numorder.sort()


def get_flag(val):
    """
    get numerical value for input flag

    Parameters
    ----------
    val: string or int
        string or int form of a flag
    """

    checkflag(val)

    try:
        # first make sure it is numerical
        3 + val
        return val
    except TypeError:
        # it must have been the string version
        return FLAG_MAP[val]


def get_flagname(val):
    """
    get name for input flag

    Parameters
    ----------
    val: string or int
        string or int form of a flag
    """

    checkflag(val)

    try:
        # if it is numerical, return name
        3 + val
        return FLAG_MAP[val]
    except TypeError:
        # it was the string
        return val


def checkflag(val):
    """
    check validity of the input flag

    Parameters
    ----------
    val: string or int
        string or int form of a flag
    """

    assert val in FLAG_MAP, 'invalid flag: %s' % val


def printflags(flags, setonly=False):
    """
    print the fraction of objects with each flag set

    Parameters
    ----------
    flags: array
        Flags array
    setonly: bool, optional
        If set, only print for those flags that are set for
        some objects
    """
    ntot = 0
    flagtot = 0
    s = '%19s %7s %8s  %s' % ('flagname', 'val', 'num', 'fraction')
    print(s)
    print('-'*len(s))
    for val in _numorder:
        if val == 0:
            continue

        flagtot |= val

        name = get_flagname(val)
        w, = np.where(flags & val != 0)
        if setonly and w.size == 0:
            continue

        ntot += w.size
        frac = w.size/flags.size
        print('%19s %7d %8d  %g' % (name, val, w.size, frac))

    print('-'*len(s))
    fractot = ntot/flags.size
    print('%19s %7d %8d  %g' % ('total', flagtot, ntot, fractot))
