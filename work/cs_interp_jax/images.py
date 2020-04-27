"""
from esheldon's espy, but modified to use matplotlib
"""

import os
import numpy
import numpy as np
import copy

import matplotlib.pyplot as plt


def view(image, ax=None, **keys):
    """
    View the image and return the biggles plot object

    If show=False just return the plot object.

    Values below zero are clipped, so pre-subtract any background as
    necessary.

    parameters
    ----------
    image or r,g,b: ndarray
        The image(s) as a 2-d array or
    nonlinear:
        Non-linear scale for an asinh scaling.  If not sent a linear scale is
        used. See the asinh_scale() function.  For asinh scaling you must scale
        your image so that it does not saturate.
    autoscale:
        For linear scaling, re-scale the image so that the maximum value is
        1.0.  This guarantees all pixels are shown without saturation.

        For asinh scaling you are responsible for performing such scalings
        Subtract this value from the image and clip values below zero.
    xlabel: string
        Label for the X axis
    ylabel: string
        Label for the Y axis
    transpose: bool
        Transpose the image.  Default False.
    ax : matpotlib axis
        The axis to plot on. If None, then a new one is created

    Returns
    -------
    ax : matplotlib axis
        The axis with the plot
    """

    # we need to transpose for biggles to display properly
    trans = keys.pop('transpose', True)

    if len(image.shape) == 2:
        im = scale_image(image, **keys)
        if trans:
            im = im.transpose()
    else:
        # for 3 d we need to trust the input to
        # be properly scaled and transposed
        im = image

    if ax is None:
        ax = plt.gca()

    if 'title' in keys:
        ax.set_title(keys['title'])

    ax.imshow(im)

    if 'xrange' in keys:
        ax.set_xlim(*keys['xrange'])
    if 'yrange' in keys:
        ax.set_ylim(*keys['yrange'])

    # make sure the pixels look square
    ax.set_aspect('equal')

    if 'xlabel' in keys:
        ax.set_xlabel(keys['xlabel'])
    if 'ylabel' in keys:
        ax.set_ylabel(keys['ylabel'])

    return ax


def bytescale(im):
    """
    The input should be between [0,1]

    output is [0,255] in a unsigned byte array
    """
    imout = (im*255).astype('u1')
    return imout


def write_image(filename, image, **keys):
    """
    Write an image to an image file.

    the image must be bytescaled between [0,255] and by of type 'u1'.  See
    the scale_image() and bytescale() functions.

    The file type, compression scheme, etc are determined by the file name.
    Extra keywords such as quality for jpegs are passed along.
    """
    from PIL import Image

    pim = Image.fromarray(image)

    fname = os.path.expandvars(filename)
    fname = os.path.expanduser(fname)
    pim.save(fname, **keys)


def compare_images(im1, im2, **keys):
    skysig = keys.get('skysig', None)
    dof = keys.get('dof', None)
    cross_sections = keys.get('cross_sections', True)

    color1 = keys.get('color1', 'blue')
    color2 = keys.get('color2', 'orange')
    colordiff = keys.get('colordiff', 'red')

    ncol = 3
    if cross_sections:
        nrow = 2
    else:
        nrow = 1

    label1 = keys.get('label1', 'im1')
    label2 = keys.get('label2', 'im2')

    cen = keys.get('cen', None)
    if cen is None:
        cen = [(im1.shape[0]-1)/2., (im1.shape[1]-1)/2.]

    labelres = '%s-%s' % (label1, label2)

    if im1.shape != im2.shape:
        raise ValueError("images must be the same shape")

    resid = im1 - im2

    width = keys.get("width", 8)
    height = keys.get("height", width * (nrow / ncol))
    fig, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(width, height),
        squeeze=False,
    )

    tkeys = copy.deepcopy(keys)
    tkeys["ax"] = axs[0, 0]
    view(im1, **tkeys)
    axs[0, 0].set_title(label1)

    tkeys["ax"] = axs[0, 1]
    view(im2, **tkeys)
    axs[0, 1].set_title(label2)

    tkeys['nonlinear'] = None
    tkeys["ax"] = axs[0, 2]
    tkeys['min'] = resid.min()
    tkeys['max'] = resid.max()
    view(resid, **tkeys)
    axs[0, 2].set_title(labelres)

    if skysig is None:
        skysig = 1.0
    if dof is None:
        dof = im1.size
    chi2per = (resid**2).sum()/skysig**2/dof
    axs[0, 2].set_title('chi^2/dof: %0.2f' % chi2per)

    # cross-sections
    if cross_sections:
        cen0 = int(cen[0])
        cen1 = int(cen[1])
        im1rows = im1[:, cen1]
        im1cols = im1[cen0, :]
        im2rows = im2[:, cen1]
        im2cols = im2[cen0, :]
        resrows = resid[:, cen1]
        rescols = resid[cen0, :]

        axs[1, 0].plot(
            im1rows,
            color=color1,
            label=label1,
            drawstyle='steps-mid',
        )
        axs[1, 0].plot(
            im2rows,
            color=color2,
            label=label2,
            drawstyle='steps-mid',
        )
        axs[1, 0].plot(
            resrows,
            color=colordiff,
            label=labelres,
            drawstyle='steps-mid',
        )
        axs[1, 0].set_xlabel("center rows")

        axs[1, 1].plot(
            im1cols,
            color=color1,
            label=label1,
            drawstyle='steps-mid',
        )
        axs[1, 1].plot(
            im2cols,
            color=color2,
            label=label2,
            drawstyle='steps-mid',
        )
        axs[1, 1].plot(
            rescols,
            color=colordiff,
            label=labelres,
            drawstyle='steps-mid',
        )
        axs[1, 1].set_xlabel("center cols")

        if keys.get("masked_image", None) is not None:
            masked_im = keys.get("masked_image")
            tkeys = copy.deepcopy(keys)
            tkeys["ax"] = axs[1, 2]
            view(masked_im, **tkeys)

    axs[0, 0].set_title(label1)


def _get_max_image(im1, im2, im3):
    maximage = im1.copy()

    w = np.where(im2 > maximage)
    if w[0].size > 1:
        maximage[w] = im2[w]

    w = np.where(im3 > maximage)
    if w[0].size > 1:
        maximage[w] = im3[w]

    return maximage


def _fix_hard_satur(r, g, b, satval):
    """
    Clip to satval but preserve the color
    """

    # make sure you send scales such that this occurs at
    # a reasonable place for your images

    maximage = _get_max_image(r, g, b)

    w = np.where(maximage > satval)
    if w[0].size > 1:
        # this preserves color
        fac = satval/maximage[w]
        r[w] *= fac
        g[w] *= fac
        b[w] *= fac
        maximage[w] = satval

    return maximage


def _fix_rgb_satur(r, g, b, fac):
    """
    Fix the factor so we don't saturate the
    RGB image (> 1)

    maximage is the
    """
    maximage = _get_max_image(r, g, b)

    w = np.where((r*fac > 1) | (g*fac > 1) | (b*fac > 1))
    if w[0].size > 1:
        # this preserves color
        fac[w] = 1.0/maximage[w]


def get_color_image(imr, img, imb, **keys):
    """
    Create a color image.

    The idea here is that, after applying the asinh scaling, the color image
    should basically be between [0,1] for all filters.  Any place where a value
    is > 1 the intensity will be scaled back in all but the brightest filter
    but color preserved.

    In other words, you develaop a set of pre-scalings the images so that after
    multiplying by

        asinh(I/nonlinear)/(I/nonlinear)

    the numbers will be mostly between [0,1].  You can send scales using the
    scale= keyword

    It can actually be good to have some color saturation so don't be too
    agressive.  You'll have to play with the numbers for each image.

    Note also the image is clipped at zero.

    TODO:
        Implement a "saturation" level in the raw image values as in
        djs_rgb_make.  Even better, implement an outside function to do this.
    """

    nonlinear = keys.get('nonlinear', 1.0)
    scales = keys.get('scales', None)
    satval = keys.get('satval', None)
    clip = keys.get('clip', None)

    r = imr.astype('f4')
    g = img.astype('f4')
    b = imb.astype('f4')

    if clip is not None:
        r.clip(clip, r.max(), r)
        g.clip(clip, g.max(), g)
        b.clip(clip, b.max(), b)

    if scales is not None:
        r *= scales[0]
        g *= scales[1]
        b *= scales[2]

    if satval is not None:
        # note using rescaled images so the satval
        # means the same thing (e.g. in terms of real flux)
        _fix_hard_satur(r, g, b, satval)

    # average images and divide by the nonlinear factor
    fac = 1./nonlinear/3.
    III = fac*(r + g + b)

    # make sure we don't divide by zero
    # due to clipping, average value is zero only if all are zero
    w = np.where(III <= 0)
    if w[0].size > 0:
        III[w] = 1./3.  # value doesn't matter images are zero

    f = np.arcsinh(III)/III

    # limit to values < 1
    # make sure you send scales such that this occurs at
    # a reasonable place for your images
    _fix_rgb_satur(r, g, b, f)

    R = r*f
    G = g*f
    B = b*f

    st = R.shape
    colorim = np.zeros((st[0], st[1], 3))

    colorim[:, :, 0] = R[:, :]
    colorim[:, :, 1] = G[:, :]
    colorim[:, :, 2] = B[:, :]

    return colorim


def scale_image(im, **keys):
    nonlinear = keys.get('nonlinear', None)
    if nonlinear is not None:
        return asinh_scale(im, nonlinear)
    else:
        return linear_scale(im, **keys)


def linear_scale(im, **keys):
    autoscale = keys.get('autoscale', True)

    III = im.astype('f4')

    if autoscale:
        maxval = np.abs(III).max()
        if maxval != 0.0:
            III *= (1.0/maxval)

    # III.clip(0.0, 1.0, III)

    return III


def asinh_scale(im, nonlinear):
    """
    Scale the image using and asinh stretch

        I = image*f
        f=asinh(image/nonlinear)/(image/nonlinear)

    Values greater than 1.0 after this scaling will be shown as white, so you
    are responsible for pre-scaling your image.  Values < 0 are clipped.

    parameters
    ----------
    image:
        The image.
    nonlinear: keyword
        The non-linear scale.
    """
    III = im.astype('f4')

    III *= (1./nonlinear)

    # make sure we don't divide by zero
    w = np.where(III <= 0)
    if w[0].size > 0:
        III[w] = 1.  # value doesn't matter since images is zero

    f = np.arcsinh(III)/III

    imout = im*f

    imout.clip(0.0, 1.0, imout)

    return imout
