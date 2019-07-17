"""
Some util functions to plot the figures that appear on the ISMIR paper.

#############

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Adapted on July 05, 2019
by Stefan Lattner

Sony CSL Paris, France
"""

__author__ = "Oriol Nieto"
__copyright__ = "Copyright 2013, Music and Audio Research Lab (MARL)"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "oriol@nyu.edu"

import numpy as np
import pylab as plt
from . import utils


def plot_segments(X, segments):
    """Plots the segments on top of half of the self similarity matrix X."""
    np.save("X.npy", X)
    np.save("segments.npy", segments)
    for s in segments:
        line_strength = 3
        np.fill_diagonal(X[s[0]:s[1], s[2]:s[3]], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] + 1:s[3] + 1], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] + 2:s[3] + 2], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] + 3:s[3] + 3], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] + 4:s[3] + 4], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] - 1:s[3] - 1], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] - 2:s[3] - 2], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] - 3:s[3] - 3], line_strength)
        np.fill_diagonal(X[s[0]:s[1], s[2] - 4:s[3] - 4], line_strength)

    offset = 15
    for i in xrange(X.shape[0]):
        if i + offset < X.shape[0]:
            for j in xrange(i + offset):
                    X[i, j] = 0
        else:
            X[i, :] = 0

    plt.figure(figsize=(6, 6))
    plt.imshow(X, interpolation="nearest", cmap=plt.cm.gray_r)
    plt.xlabel("Time Frames")
    plt.ylabel("Time Frames")
    plt.tight_layout()
    plt.savefig("paths-found.pdf")
    plt.show()


def plot_ssm(X):
    """Plots a self-similarity matrix, as it appears on the ISMIR paper.

    Parameters
    ----------
    X : np.array((N,N))
        Self-similarity matrix
    """
    plt.figure(figsize=(6, 6))
    Y = (X[3:, 3:] + X[2:-1, 2:-1] + X[1:-2, 1:-2] + X[:-3, :-3]) / 3.
    plt.imshow((1 - Y), interpolation="nearest", cmap=plt.cm.gray)
    h = 1705
    m = 245.
    l = 2.0
    #plt.axvline(28 * h / m, color="k", linewidth=l)
    #plt.axvline(50 * h / m, color="k", linewidth=l)
    #plt.axvline(70 * h / m, color="k", linewidth=l)
    #plt.axvline(91 * h / m, color="k", linewidth=l)
    #plt.axvline(110 * h / m, color="k", linewidth=l)
    #plt.axvline(135 * h / m, color="k", linewidth=l)
    #plt.axvline(157 * h / m, color="k", linewidth=l)
    #plt.axvline(176 * h / m, color="k", linewidth=l)
    #plt.axvline(181 * h / m, color="k", linewidth=l)
    #plt.axvline(202 * h / m, color="k", linewidth=l)

    #plt.axhline(28 * h / m, color="k", linewidth=l)
    #plt.axhline(50 * h / m, color="k", linewidth=l)
    #plt.axhline(70 * h / m, color="k", linewidth=l)
    #plt.axhline(91 * h / m, color="k", linewidth=l)
    #plt.axhline(110 * h / m, color="k", linewidth=l)
    #plt.axhline(135 * h / m, color="k", linewidth=l)
    #plt.axhline(157 * h / m, color="k", linewidth=l)
    #plt.axhline(176 * h / m, color="k", linewidth=l)
    #plt.axhline(181 * h / m, color="k", linewidth=l)
    #plt.axhline(202 * h / m, color="k", linewidth=l)
    plt.xlabel("Time frames")
    plt.ylabel("Time frames")
    plt.savefig("SSM-euclidean-annotation.pdf")
    plt.show()


def plot_chroma(C):
    """Plots a Chromagram example, as it appears on the ISMIR paper.

    Parameters
    ----------
    C : np.array((N,12))
        Chromagram.
    """
    plt.figure(figsize=(8, 3))
    plt.imshow((1 - C.T), interpolation="nearest", aspect="auto",
               cmap=plt.cm.gray)
    plt.yticks(np.arange(12), ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F",
                               "F#", "G", "G#"))
    h = 1705
    m = 245.
    l = 2.0
    plt.axvline(28 * h / m, color="k", linewidth=l)
    plt.axvline(50 * h / m, color="k", linewidth=l)
    plt.axvline(70 * h / m, color="k", linewidth=l)
    plt.axvline(91 * h / m, color="k", linewidth=l)
    plt.axvline(110 * h / m, color="k", linewidth=l)
    plt.axvline(135 * h / m, color="k", linewidth=l)
    plt.axvline(157 * h / m, color="k", linewidth=l)
    plt.axvline(176 * h / m, color="k", linewidth=l)
    plt.axvline(181 * h / m, color="k", linewidth=l)
    plt.axvline(202 * h / m, color="k", linewidth=l)
    plt.xticks([0, C.shape[0] - 1], [0, "N"])
    plt.xlabel("Time (frames)")
    plt.tight_layout()
    plt.show()


def plot_score_examples(X):
    """Plots some examples of the score, as they appear on the ISMIR paper.

    Parameters
    ----------
    X : np.array((N,N))
        Self-similarity matrix
    """
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    plt.subplots_adjust(wspace=.05)
    props = dict(boxstyle='round', facecolor='white', alpha=0.95)
    cm = plt.cm.gray

    # Synthesized matrix
    X1 = np.zeros((12, 12))
    np.fill_diagonal(X1, 1)
    utils.compute_segment_score_omega(X1, 0, 0, 10, 0.35, 3)
    ax1.imshow(1 - X1, interpolation="nearest", cmap=cm)
    textstr = "$\sigma$(1)=1\n$\sigma$(2)=0.36\n$\sigma$(3)=0.22"
    ax1.text(5.7, 0.005, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    ax1.set_xticks(np.empty(0), np.empty(0))
    ax1.set_yticks(np.empty(0), np.empty(0))
    ax1.set_title("(a)")

    # Real matrix with an actual path
    X2 = X[359:359 + 31, 1285:1285 + 31]
    utils.compute_segment_score_omega(X, 359, 1285, 31, 0.35, 3)
    ax2.imshow(1 - X2, interpolation="nearest", cmap=cm)
    textstr = "$\sigma$(1)=-0.48\n$\sigma$(2)=0.44\n$\sigma$(3)=0.55"
    ax2.text(15.00, 0.55, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    ax2.set_xticks(np.empty(0), np.empty(0))
    ax2.set_yticks(np.empty(0), np.empty(0))
    ax2.set_title("(b)")

    utils.compute_segment_score(X, 500, 1100, 31, 0.35)
    utils.compute_segment_score_omega(X, 500, 1100, 31, 0.35, 3)
    X3 = X[500:500 + 31, 1100:1100 + 31]
    ax3.imshow(1 - X3, interpolation="nearest", cmap=cm)
    textstr = "$\sigma$(1)=-0.46\n$\sigma$(2)=0.21\n$\sigma$(3)=0.32"
    ax3.text(15.00, 0.55, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    ax3.set_xticks(np.empty(0), np.empty(0))
    ax3.set_yticks(np.empty(0), np.empty(0))
    ax3.set_title("(c)")

    plt.tight_layout()
    plt.show()
