"""
Created on Mar 5, 2015
Edited on April 13, 2018

@author: Stefan Lattner & Maarten Grachten

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""

import logging
import os

import numpy as np
import numpy.ma as ma
from matplotlib import use

use('agg')

import matplotlib.pyplot as plt
import PIL.Image

from complex_auto.util import to_numpy

LOGGER = logging.getLogger(__name__)


def uniquifier(seq):
    # Removes all double entries of seq
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def plot_kernels(model, epoch, out_dir):
    """
    Plots the input weights of the C-GAE

    :param model: a C-GAE instance
    :param epoch: epoch nr (int)
    :param out_dir: directory where to save the plot
    """
    filter_x = to_numpy(model.conv_x.weight)
    make_tiles(filter_x, os.path.join(out_dir, f"filtersx_ep{epoch}.png"))

    filter_y = to_numpy(model.conv_y.weight)
    make_tiles(filter_y, os.path.join(out_dir, f"filtersy_ep{epoch}.png"))


def plot_recon(model, x, y, epoch, out_dir):
    """
    Plots the reconstruction of an input batch

    :param model: a C-GAE instance
    :param batch: the batch to reconstruct
    :param epoch: epoch nr
    :param out_dir: directory where to save the plot
    """
    output = to_numpy(model(model.mapping_(x, y), y))
    make_tiles(output, os.path.join(out_dir, f"recon_ep{epoch}.png"))


def plot_mapping(model, x, y, epoch, out_dir):
    """
    Plots the top most mapping layer given an input batch

    :param model: a C-GAE instance
    :param batch: the batch to reconstruct
    :param epoch: epoch nr
    :param out_dir: directory where to save the plot
    """
    # x = cuda_variable(batch[0][None,:,:,:])
    # y = cuda_variable(batch[1][None,:,:,:])
    output = np.transpose(to_numpy(model.mapping_(x, y)), axes=(0, 3, 2, 1))
    make_tiles(output, os.path.join(out_dir, f"mapping_ep{epoch}.png"))


def plot_data(data, epoch, out_dir):
    """
    Plots input data

    :param data: a data batch to plot
    :param epoch: epoch nr
    :param out_dir: directory where to save the plot
    """
    make_tiles(to_numpy(data), os.path.join(out_dir, f"data_ep{epoch}.png"))


def plot_histograms(model, x, y, epoch, out_dir):
    """
    Plots some histograms

    :param model: a C-GAE instance
    :param batch: the batch to use for computing histograms
    :param epoch: epoch nr
    :param out_dir: directory where to save the histograms
    """
    plot_hist(to_numpy(x), f"data_ep{epoch}",
              os.path.join(out_dir, f"hist_data_ep{epoch}.png"))

    mapping = to_numpy(model.mapping_(x, y))
    plot_hist(mapping, f"mapping_ep{epoch}",
              os.path.join(out_dir, f"hist_map_ep{epoch}.png"))
    output = to_numpy(model(model.mapping_(x, y), y))
    plot_hist(output, f"recon_ep{epoch}",
              os.path.join(out_dir, f"hist_recon_ep{epoch}.png"))

    weight_x = to_numpy(model.conv_x.weight)
    plot_hist(weight_x, f"weightx_ep{epoch}",
              os.path.join(out_dir, f"hist_weightx_ep{epoch}.png"))
    weight_y = to_numpy(model.conv_y.weight)
    plot_hist(weight_y, f"weighty_ep{epoch}",
              os.path.join(out_dir, f"hist_weighty_ep{epoch}.png"))


""" General plotting functions """


def plot_colormap(values, title, filename):
    """
    Save a colormap of values titled with title in file filename.
    """
    # LOGGER.debug("img_size self-sim: {0}".format(values.shape))
    plt.imshow(values)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def make_tiles(tiles_in_parts, out_file,
               vp_separator_color=(100, 150, 100),
               tile_separator_color=(25, 75, 25),
               unit=True):
    """
    Plots tiles in a grid (performs a normalization!).
    Useful for plotting the weights of a neural network or data instances.

    :param tiles_in_parts: tiles to plot
    :param out_file: output filename
    """
    tiles = [make_tile(tile_parts, vp_separator_color, unit) for tile_parts in
             tiles_in_parts]

    assert len(tiles) > 0
    tile_height, tile_width, _ = tiles[0].shape

    n_tiles = len(tiles)
    n_tiles_horz = max(1, min(n_tiles, int(
        ((tile_height * n_tiles) / tile_width) ** .5)))
    n_tiles_vert = int(np.ceil(n_tiles / float(n_tiles_horz)))

    pane = np.zeros(((n_tiles_vert * (tile_height + 1)) - 1,
                     (n_tiles_horz * (tile_width + 1)) - 1,
                     3), dtype=np.uint8)

    offset_horz = 0
    offset_vert = 0

    for i, tile in enumerate(tiles):
        if i % n_tiles_horz == 0 and n_tiles_vert > 1:
            pane[offset_vert + tile_height, :, :] = tile_separator_color
            if i > 0:
                offset_horz = 0
                offset_vert += tile_height + 1

        pane[offset_vert: offset_vert + tile_height,
        offset_horz: offset_horz + tile_width, :] = tile
        offset_horz += tile_width + 1
        if offset_horz - 1 < pane.shape[1]:
            pane[:, offset_horz - 1, :] = tile_separator_color

    filt_img = PIL.Image.fromarray(pane, mode="RGB")
    filt_img.save(out_file)
    return pane


def make_tile(tile_parts, vp_separator_color=(255, 0, 0), unit=True):
    """
    This function could do a simple hstack, if it weren't for the
    separation lines that we want between viewpoints. Furthermore,
    we need to rescale the values of the viewpoints jointly, but do
    not want the separation lines to interfere with the
    rescaling. Therefore, we use a masked array.

    NOTE: the tile is transposed after it is constructed. This means
    that with respect to the plots displaying the tiles, the meaning
    of width and height in this function are swapped
    """

    # compute shape of tile, including separation lines
    shapes = np.array([x.shape for x in tile_parts], np.int)
    # assume the viewpoints have all the same size in the first
    # dimension (the ngram size)
    assert np.std(shapes[:, 0]) == 0

    # tile_height equals the ngram size
    tile_height = shapes[0, 0]
    # tile_width equals the sum of the viewpoint sizes plus (nr of
    # viewpoints) - 1 for the viewpoint seperation lines

    tile_width = np.sum(shapes[:, 1]) + len(tile_parts) - 1

    # create an empty masked array the size of the tile (data and mask
    # will be set)
    tile = ma.masked_array(np.empty((tile_height, tile_width)))

    # copy the viewpoint data to their respective locations in the
    # tile, set a mask on the line after each viewpoint (except the
    # last)
    offset = 0
    for part in tile_parts:

        # assign viewpoint data
        tile[:, offset: offset + part.shape[1]] = part
        offset += part.shape[1] + 1

        # we are not at the last part
        if offset < tile_width:
            # set mask
            tile[:, offset - 1] = ma.masked

    # scale the values to 0-255, round, and convert to uint8
    tile = np.round(scale_to_unit_interval(tile) * 255).astype(np.uint8)

    # duplicate tile for RGB data
    tr = tile.reshape(list(tile.shape) + [1]).repeat(3, axis=2)

    offset = 0
    # set the separation lines between the viewpoints (overrides
    # the mask)
    for part in tile_parts:
        offset += part.shape[1] + 1
        if offset < tile_width:
            tr[:, offset - 1, :] = vp_separator_color

    return tr.transpose((1, 0, 2))


def scale_to_unit_interval(ndar, eps=1e-8):
    """
    Scales all values in the ndarray ndar to be between 0 and 1
    """
    ndar = ndar.copy()
    max_val = np.max((np.abs(ndar).max(), 1e-4))
    ndar /= (2 * max_val + eps)
    ndar += 0.5
    return ndar


def plot_hist(values, title, filename, bins=50):
    """
    Save a histogram of values titled with title in file filename.
    """
    plt.clf()
    plt.hist(values.flatten(), bins=bins)
    plt.xlabel('Value')
    plt.ylabel('Amount')
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_curve(curve, title, fn):
    '''
    Save a plot of a function (curve).
    '''
    plt.clf()
    plt.plot(curve)
    plt.title(title)
    plt.savefig(fn)
    plt.close()


def plot_fs_2d(data_set, labels=None, tooltip=None, size=(10, 10),
               save_fn=None,
               interactive=False, cmap=None, show_centroids=True,
               label_centroids=True, style_data='x', style_centroid='o',
               color_dict = None):
    """
    Plot a given dataset and utilizes labels.
    Plots a given 2d dataset and uses different colors for datapoints belonging
    to different labels.
    Displays a centroid for each label cluster and names it with the label string.

    Parameters
    ----------

    data_set: array-like
        2D matrix of size M x N which is to be points

    labels: array-like, optional
        1D array of size M, preferrable of dtype string

    tooltip: array-like, optional
        1D array of size M, used as tooltip annotation for points

    size: tuple, optional
        The size of the plot

    save_fn: string, optional
        If not None, a file will be saved to that path

    cmap: matplotlib.cm, optional
        Colormap which will be used for plotting datapoints of each label in
        a different color

    show_centroids: boolean, optional
        Determines if centroids should be points

    label_centroids: boolean, optional
        Determines if centroids should be labelled

    Returns
    -------

    figure
        The figure. Use figure.save()

    """

    def color_iterator(cmap, n):
        return iter(cmap(np.linspace(0,1,n)))

    assert len(data_set[0]) == 2  # Input matrix has to be 2d.

    if interactive:
        plt.interactive(True)

    if cmap == None:
        cmap = plt.get_cmap("Paired")
    else:
        cmap = plt.get_cmap(cmap)

    data_set = np.array(data_set)

    fig = plt.figure(figsize=size)
    ax = fig.gca()
    points = None

    if labels is not None:
        labels = np.array(labels)
        labels_single = uniquifier(labels)
        colors = color_iterator(cmap, len(labels_single))

        for label in labels_single:
            col = next(colors) if color_dict is None else color_dict[label]
            data_xy = (data_set[labels == label,0], data_set[labels == label,1])
            points = ax.plot(data_xy[0], data_xy[1], style_data, color=col, markersize = 2)

        colors = color_iterator(cmap, len(labels_single))
        if show_centroids:
            for label in labels_single:
                col = next(colors) if color_dict is None else color_dict[label]
                mean_xy = (np.mean(data_set[labels==label,0]),
                           np.mean(data_set[labels==label,1]))
                ax.plot(mean_xy[0], mean_xy[1], style_centroid, color=col)
                bbox_props = dict(boxstyle="round,pad=0.3", fc=col, ec="grey",
                                  lw=1, alpha=0.9)
                if label_centroids:
                    plt.annotate(label, xy = mean_xy, xytext = (-15, 7),
                                 textcoords = 'offset points', bbox=bbox_props)
    else:
        col = cmap(0)
        points = ax.plot(data_set[:,0], data_set[:,1], style_data, color=col)

    if save_fn is not None:
        LOGGER.debug('Saving 2D feature space projection to {0}'.format(save_fn))
        plt.savefig(save_fn)

    if interactive:
        raise Exception("Interactive plot deactivated (no libs installed)")
        desc = [unicode(l, 'utf-8') for l in tooltip]
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], desc)
        mpld3.plugins.connect(fig, tooltip)
        mpld3.show()

    # while not plt.waitforbuttonpress(timeout=-1):
    #     pass

    if interactive:
        plt.interactive(False)

    return fig