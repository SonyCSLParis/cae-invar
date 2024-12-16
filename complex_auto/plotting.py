
"""
Created on July 05, 2019

@author: Stefan Lattner

Sony CSL Paris, France

"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from complex_auto.plot import plot_hist, make_tiles
from complex_auto.util import cuda_variable, to_numpy


def plot_train_state_2d(loss_curve_eval, loss_curve_train, model, x, y,
                        epoch, out_dir, length_ngram):
    model.eval()
    x = cuda_variable(x)
    y = cuda_variable(y)
    # calculate mapping of untransposed data
    amp_x, phase_x = model(x)
    amp_y, phase_y = model(y)

    recon_y = model.backward(amp_x, phase_y)
    recon_x = model.backward(amp_y, phase_x)

    make_tiles(to_numpy(recon_y).reshape(recon_y.shape[0], 1, -1,
                                         length_ngram),
               os.path.join(out_dir, f"recon_y{epoch}.png"))
    make_tiles(to_numpy(recon_x).reshape(recon_x.shape[0], 1, -1,
                                         length_ngram),
               os.path.join(out_dir, f"recon_x{epoch}.png"))
    plot_hist(to_numpy(recon_x), f"recon_x_hist_ep{epoch}",
              os.path.join(out_dir, f"recon_x_hist_ep{epoch}.png"))
    plot_hist(to_numpy(recon_y), f"recon_y_hist_ep{epoch}",
              os.path.join(out_dir, f"recon_y_hist_ep{epoch}.png"))

    half_weight = model.layer.weight.shape[0] // 2
    make_tiles(to_numpy(model.layer.weight[:half_weight]).reshape(len(
                                        model.layer.weight[:half_weight]),
                                                    1, -1,
                                                    length_ngram),
               os.path.join(out_dir, f"filters_x{epoch}.png"))
    make_tiles(to_numpy(model.layer.weight[half_weight:]).reshape(len(
                                    model.layer.weight[half_weight:]),
                                                    1, -1,
                                                    length_ngram),
               os.path.join(out_dir, f"filters_y{epoch}.png"))
    plot_hist(to_numpy(model.layer.weight[:half_weight]).reshape(len(
                                        model.layer.weight[:half_weight]),
                                                    1, -1,
                                                    length_ngram),
                                        f"filters_x_hist_ep{epoch}",
              os.path.join(out_dir, f"filters_x_hist_ep{epoch}.png"))
    plot_hist(to_numpy(model.layer.weight[half_weight:]).reshape(len(
                                    model.layer.weight[half_weight:]),
                                                    1, -1,
                                                    length_ngram),
                                        f"filters_y_hist_ep{epoch}",
              os.path.join(out_dir, f"filters_y_hist_ep{epoch}.png"))

    # make_tiles(to_numpy(model.layer.weight).reshape(len(
    #     model.layer.weight),
    #     1, -1,
    #     length_ngram),
    #     os.path.join(out_dir, f"filters_x{epoch}.png"))

    make_tiles(to_numpy(amp_x)[:, None, None, :], os.path.join(out_dir,
                                             f"ampx_{epoch}.png"))
    plot_hist(to_numpy(amp_x), f"ampx_hist_ep{epoch}",
              os.path.join(out_dir, f"ampx_hist_ep{epoch}.png"))
    make_tiles(to_numpy(amp_y)[:, None, None, :], os.path.join(out_dir,
                                             f"ampy_{epoch}.png"))
    plot_hist(to_numpy(amp_y), f"ampy_hist_ep{epoch}",
              os.path.join(out_dir, f"ampy_hist_ep{epoch}.png"))

    make_tiles(to_numpy(phase_x)[:, None, None, :], os.path.join(out_dir,
                                                               f"phasex_{epoch}.png"))
    plot_hist(to_numpy(phase_x), f"phasex_hist_ep{epoch}",
              os.path.join(out_dir, f"phasex_hist_ep{epoch}.png"))
    make_tiles(to_numpy(phase_y)[:, None, None, :], os.path.join(out_dir,
                                                               f"phasey_{epoch}.png"))
    plot_hist(to_numpy(phase_y), f"phasey_hist_ep{epoch}",
              os.path.join(out_dir, f"phasey_hist_ep{epoch}.png"))

    plot_hist(to_numpy(x.data), f"input_hist_ep{epoch}",
              os.path.join(out_dir, f"input_hist_ep{epoch}.png"))
    plot_hist(to_numpy(y.data), f"target_hist_ep{epoch}",
              os.path.join(out_dir, f"target_hist_ep{epoch}.png"))

    input_np = to_numpy(x.data)
    make_tiles(input_np.reshape(x.shape[0], 1, -1, length_ngram),
               os.path.join(out_dir, f"input_{epoch}.png"))
    target_np = to_numpy(y.data)
    make_tiles(target_np.reshape(y.shape[0], 1, -1, length_ngram),
               os.path.join(out_dir, f"target_{epoch}.png"))
    plt.clf()
    plt.plot(loss_curve_train)
    plt.plot(loss_curve_eval)
    plt.savefig(
        os.path.join(out_dir, f"loss_curve_{epoch}.png"))


def plot_train_state_1d(loss_curve_eval, loss_curve_train, model, x, y,
                        epoch, out_dir, length_ngram):
    model.eval()
    x = cuda_variable(x)
    y = cuda_variable(y)
    # calculate mapping of untransposed data
    amp_x, phase_x = model(x)
    amp_y, phase_y = model(y)

    recon_y = model.backward(amp_x, phase_y)
    recon_x = model.backward(amp_y, phase_x)

    plot_audiobatch(recon_y[:20, None, :].detach().cpu(),
                    os.path.join(out_dir, f"recon_y{epoch}.png"))
    plot_audiobatch(recon_x[:20, None, :].detach().cpu(),
                    os.path.join(out_dir, f"recon_x{epoch}.png"))
    plot_audiobatch(y[:20, None, :].detach().cpu(),
                    os.path.join(out_dir, f"input_y_sig{epoch}.png"))
    plot_audiobatch(x[:20, None, :].detach().cpu(),
                    os.path.join(out_dir, f"input_x_sig{epoch}.png"))
    plot_hist(to_numpy(recon_x), f"recon_x_hist_ep{epoch}",
              os.path.join(out_dir, f"recon_x_hist_ep{epoch}.png"))
    plot_hist(to_numpy(recon_y), f"recon_y_hist_ep{epoch}",
              os.path.join(out_dir, f"recon_y_hist_ep{epoch}.png"))

    make_tiles(to_numpy(model.layer.weight).reshape(len(
                                                    model.layer.weight),
                                                    1, -1,
                                                    length_ngram),
               os.path.join(out_dir, f"filters_x{epoch}.png"))
    make_tiles(to_numpy(model.layer.weight).reshape(len(
                                                    model.layer.weight),
                                                    1, -1,
                                                    length_ngram),
               os.path.join(out_dir, f"filters_y{epoch}.png"))
    plot_audiobatch(model.layer.weight[:20, None, :].detach().cpu(),
                    os.path.join(out_dir, f"filters_sig_real{epoch}.png"))
    half = model.layer.weight.shape[0] // 2
    plot_audiobatch(model.layer.weight[half:half+20, None, :].detach().cpu(),
                    os.path.join(out_dir, f"filters_sig_compl{epoch}.png"))

    make_tiles(to_numpy(amp_x)[:, None, None, :], os.path.join(out_dir,
                                             f"ampx_{epoch}.png"))
    plot_hist(to_numpy(amp_x), f"ampx_hist_ep{epoch}",
              os.path.join(out_dir, f"ampx_hist_ep{epoch}.png"))
    make_tiles(to_numpy(amp_y)[:, None, None, :], os.path.join(out_dir,
                                             f"ampy_{epoch}.png"))
    plot_hist(to_numpy(amp_y), f"ampy_hist_ep{epoch}",
              os.path.join(out_dir, f"ampy_hist_ep{epoch}.png"))

    make_tiles(to_numpy(phase_x)[:, None, None, :], os.path.join(out_dir,
                                                               f"phasex_{epoch}.png"))
    plot_hist(to_numpy(phase_x), f"phasex_hist_ep{epoch}",
              os.path.join(out_dir, f"phasex_hist_ep{epoch}.png"))
    make_tiles(to_numpy(phase_y)[:, None, None, :], os.path.join(out_dir,
                                                               f"phasey_{epoch}.png"))
    plot_hist(to_numpy(phase_y), f"phasey_hist_ep{epoch}",
              os.path.join(out_dir, f"phasey_hist_ep{epoch}.png"))

    plot_hist(to_numpy(x.data), f"input_hist_ep{epoch}",
              os.path.join(out_dir, f"input_hist_ep{epoch}.png"))
    plot_hist(to_numpy(y.data), f"target_hist_ep{epoch}",
              os.path.join(out_dir, f"target_hist_ep{epoch}.png"))

    input_np = to_numpy(x.data)
    make_tiles(input_np.reshape(x.shape[0], 1, -1, length_ngram),
               os.path.join(out_dir, f"input_{epoch}.png"))
    target_np = to_numpy(y.data)
    make_tiles(target_np.reshape(y.shape[0], 1, -1, length_ngram),
               os.path.join(out_dir, f"target_{epoch}.png"))
    plt.clf()
    plt.plot(loss_curve_train)
    plt.plot(loss_curve_eval)
    plt.savefig(
        os.path.join(out_dir, f"loss_curve_{epoch}.png"))


def plot_audiobatch(batch: torch.Tensor, fn, num_example: int=30,
                    verbose: bool=False):
    # by Stephane Rivaud

    plt.clf()
    audio_list = [(audio.squeeze(), 22050) for audio in batch]

    # determine the number of rows and columns
    ncols = int(math.sqrt(len(audio_list)))
    nrows = math.ceil(len(audio_list) / ncols)

    # plotting files
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    for k in range(len(audio_list)):
        audio, sr = audio_list[k]
        i, j = k // ncols, k % ncols
        x = torch.arange(audio.size(0), dtype=torch.float32) / sr
        ax[i, j].plot(x.numpy(), audio.numpy(), linewidth=1)
        #ax[i, j].set_xlabel('Time (s)')
        #ax[i, j].set_ylabel('Amplitude')
        #ax[i, j].set_title(f'Sample {k}')
    #plt.show()
    plt.savefig(fn)
    #plt.savefig(fn+".pdf")
