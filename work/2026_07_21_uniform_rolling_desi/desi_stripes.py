import numpy as np


def get_desi_stripes():
    # code for DESI stripes from Anand Raichoor
    names = np.array([
        "NGC-1", "NGC-2", "NGC-3", "NGC-4", "NGC-5", "NGC-6", "NGC-7", "NGC-8", "NGC-9",
        "SGC-1", "SGC-2", "SGC-3", "SGC-4", "SGC-5", "SGC-6", "SGC-7", "SGC-8", "SGC-9",
    ])
    surveys = np.array([
        "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "Niji+VST", "Niji+VST", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS",
        "Niji+VST", "Niji+VST", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS",
    ])
    # original years
    # yrs = np.array([
    #     4, 3, 2, 2, 4, 3, 1, 1, 5,
    #     4, 3, 3, 2, 2, 1, 1, 4, 5,
    # ])
    # swapped times for NGC "Niji+VST", "Niji+VST" fields w/ "POLU+IBIS" fields
    # reordered stripes for POU+IBIS to make bigger coherent areas
    yrs = np.array([
        4, 4, 3, 3, 2, 2, 1, 1, 5,
        4, 3, 3, 2, 2, 1, 1, 4, 5,
    ])
    ramins = np.array([
        121.01, 122.33, 123.69, 125.11, 126.60, 128.16, 129.81, 131.56, 133.43,
        315.57, 313.82, 312.19, 310.66, 309.21, 307.84, 306.53, 305.29, 304.10,
    ])
    ramaxs = np.array([
        255.71, 254.40, 253.00, 251.52, 249.93, 228.30, 225.58, 222.41, 218.59,
        44.37, 47.15, 49.56, 51.67, 53.53, 55.18, 56.64, 57.93, 59.06,
    ])
    decmins = np.array([
        11.375, 8.875, 5.875, 3.125, 0.375, -2.375, -5.375, -8.125, -10.875,
        1.625, -1.375,  -4.125, -6.875, -9.625, -12.375, -15.375, -18.125, -20.875,
    ])
    decmaxs = np.array([
        15.125, 12.375, 9.375, 6.625, 3.875, 1.125, -1.625, -4.625, -7.375,
        5.125, 2.375, -0.625, -3.375, -6.125, -8.875, -11.625, -14.625, -17.375,
    ])
    return {
        "names": names,
        "surveys": surveys,
        "yrs": yrs,
        "ramins": ramins,
        "ramaxs": ramaxs,
        "decmins": decmins,
        "decmaxs": decmaxs
    }


def is_in_ra_range(ra, low, high):
    _low = low % (2.0 * np.pi)
    _high = high % (2.0 * np.pi)
    if _low <= _high:
        return (ra >= _low) & (ra <= _high)
    else:
        return (ra >= _low) | (ra <= _high)

