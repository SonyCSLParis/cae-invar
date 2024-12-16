#!/usr/bin/env python

"""
Created on July 05, 2019

@author: Stefan Lattner

Sony CSL Paris, France

"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from complex_auto.complex import Complex
from complex_auto.config import get_config, config_to_args
from complex_auto.cqt import get_cqts
from complex_auto.dataloader import DataSampler, Signal, Data
from complex_auto.plot import plot_fs_2d
from complex_auto.plotting import plot_train_state_2d, plot_train_state_1d
from complex_auto.regularize import norm_loss, equal_norm_loss
from complex_auto.transform import rPCA
from complex_auto.util import load_pyc_bz, cuda_variable, prepare_audio_inputs

LOGGER = logging.getLogger(__name__)


def plot_train_state(loss_curve_eval, loss_curve_train, model, x, y, epoch):
    if args.data_type == 'cqt':
        plot_train_state_2d(loss_curve_eval, loss_curve_train, model, x, y,
                            epoch, out_dir, args.length_ngram)
    elif args.data_type == 'mnist':
        plot_train_state_2d(loss_curve_eval, loss_curve_train, model, x, y,
                            epoch, out_dir, args.length_ngram)
    else:
        plot_train_state_1d(loss_curve_eval, loss_curve_train, model, x, y,
                            epoch, out_dir, args.length_ngram)


def create_data_loaders(data, data_eval, data_test, length_ngram, \
                        samples_epoch, batch_size, transform, emph_onset=0):
    train_loader = DataLoader(
        DataSampler(data, length_ngram=length_ngram,
                    samples_epoch=samples_epoch,
                    transform=transform,
                    emph_onset=emph_onset,
                    random_pairs=False,
                    shifts=args.shifts,
                    scales=args.scales
                    ),
        batch_size=batch_size,
        shuffle=True, **kwargs)

    eval_loader = DataLoader(
        DataSampler(data_eval, length_ngram=length_ngram,
                    samples_epoch=samples_epoch, shuffle=False,
                    transform=transform, emph_onset=emph_onset,
                    random_pairs=False,
                    shifts=args.shifts,
                    scales=args.scales
                    ),
        batch_size=500,
        shuffle=True, **kwargs)

    test_loader = DataLoader(
        DataSampler(data_test, length_ngram=length_ngram,
                    samples_epoch=samples_epoch,
                    shuffle=False,
                    transform=transform, emph_onset=emph_onset,
                    random_pairs=False,
                    shifts=[0, 0],
                    scales=[0, 0]
                    ),
        batch_size=1,
        shuffle=False, **kwargs)
    return train_loader, eval_loader, test_loader


def create_data_sig(block_size=5000, refresh_cache=True):
    with open(args.input_files, 'r') as f:
        files = f.readlines()

    files_eval = files

    train_loader = torch.utils.data.DataLoader(
        Signal(files, trg_shift=0,
               block_size=block_size,
               refresh_cache=refresh_cache,
               random_shift=args.shifts[0],
               samples_epoch=args.samples_epoch),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    eval_loader = torch.utils.data.DataLoader(
        Signal(files_eval, trg_shift=0,
               block_size=block_size,
               refresh_cache=False,
               random_shift=args.shifts[0],
               samples_epoch=args.samples_epoch),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader, eval_loader


def create_data_mnist(data, data_eval, batch_size):
    train_loader = DataLoader(
        Data(data[:, 0], data[:, 1], standardize=True),
        batch_size=batch_size,
        shuffle=True, **kwargs)

    eval_loader = DataLoader(
        Data(data_eval[:, 0], data_eval[:, 1], standardize=True),
        batch_size=batch_size,
        shuffle=True, **kwargs)
    return train_loader, eval_loader


def plot_pca(loader, fn="pca_trans.png"):
    angles_diff = []
    instance_nrs = []
    labels = []
    mags_all_x = []
    mags_all_y = []
    model.cpu()
    model.eval()
    for batch_idx, (x, y, transform, instance_nr, distance) in enumerate(
            loader):
        mags_x, angles_x = model(x)
        mags_y, angles_y = model(y)
        mags_all_x.append(mags_x.detach())
        mags_all_y.append(mags_y.detach())
        angles_diff.append((angles_x - angles_y).detach())
        instance_nrs.append(instance_nr)
        labels.append(distance)
        if batch_idx == 50:
            break

    angles = np.vstack(angles_diff)

    mags_all_x = np.vstack(mags_all_x)
    mags_all_y = np.vstack(mags_all_y)
    labels = np.hstack(labels)

    mags_prod = mags_all_x * mags_all_y
    quartile = mags_prod.copy()
    np.matrix.sort(quartile)
    quartile = quartile[:, 0]

    angles[mags_prod < quartile[:, None]] = 0
    angles = np.arcsin(np.sin(angles))

    dims = 6
    pca = rPCA(angles, M=dims)
    vals = pca.pca_transform()
    print(f"Var explained = {pca.pca_expvar()}")
    print(f"size_vals = {vals.shape}")
    for i in range(dims - 2):
        vals_dim = vals[:, i:i + 2]
        plot_fs_2d(vals_dim, labels, save_fn=os.path.join(out_dir,
                                                             f"dim{i}" + fn))
    if args.cuda:
        model.cuda()


def eval(eval_loader):
    model.eval()
    losses = []
    diffs = []
    for batch_idx, (x, y, transform, _, _) in enumerate(eval_loader):
        x = cuda_variable(x)
        y = cuda_variable(y)

        amps_x, phases_x = model(x)
        amps_y, phases_y = model(y)

        recon_y = model.backward(amps_x, phases_y)
        recon_x = model.backward(amps_y, phases_x)

        pow = 1
        loss_mse = (torch.abs(y - recon_y) ** pow).mean() + \
                   (torch.abs(x - recon_x) ** pow).mean()

        losses.append(loss_mse.item())

    # print(losses)
    return losses, diffs


def train(model, train_loader):
    """
    Trains the C-GAE for one epoch

    :param epoch: the current training epoch
    """
    model.train()
    losses = []
    diffs = []

    for batch_idx, (x, y, transform, _, _) in enumerate(train_loader):
        x = cuda_variable(x)
        y = cuda_variable(y)

        # Full model
        optimizer.zero_grad()

        amps_x, phases_x = model(x)
        amps_y, phases_y = model(y)

        recon_y = model.backward(amps_x, phases_y)
        recon_x = model.backward(amps_y, phases_x)

        normloss = cuda_variable(torch.Tensor(np.array([0])))
        eqnloss = cuda_variable(torch.Tensor(np.array([0])))

        if args.norm_loss > 0:
            normloss = norm_loss(model.layer.weight) * args.norm_loss

        if args.equal_norm_loss > 0:
            eqnloss = equal_norm_loss(model.layer.weight) * \
                      args.equal_norm_loss

        pow = args.power_loss
        loss_mse = ((y - recon_y).abs() ** pow).mean() + \
                   ((x - recon_x).abs() ** pow).mean()

        loss = loss_mse + normloss + eqnloss

        loss.backward()
        optimizer.step()

        if args.set_to_norm >= 0:
            model.set_to_norm(args.set_to_norm)

        losses.append(loss.item())

    if args.set_to_norm >= 0:
        print(f"Learned norm for all bases = {model.norm_val.item()}")

    return losses, diffs


def run_experiment(model, start_epoch):
    # Train and plot intermediate results
    loss_curve = []
    eval_curve = []
    bs = args.batch_size
    print("Starting training...")
    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        losses, precs, eval_loss = None, None, None

        losses, diffs = train(model, train_loader)

        model.eval()
        eval_loss, eval_diff = eval(eval_loader)

        loss_curve.append(np.mean(losses))
        eval_curve.append(np.mean(eval_loss))

        if epoch % 1 == 0:
            print('Finished Epoch: {}/{} ({:.0f}%)\tLoss: {:.6f}'
                  '\tEval: {:.6f}\tBatch Size: {}'.format(
                epoch, args.epochs + 1, 100. * epoch / args.epochs + 1,
                np.mean(losses), np.mean(eval_loss), bs))

        if epoch % args.plot_interval == 0:
            for batch_idx, (x, y, transform, _, _) in enumerate(train_loader):
                if batch_idx == 0:
                    x = cuda_variable(x)
                    y = cuda_variable(y)
                    plot_train_state(loss_curve, eval_curve, model, x, y,
                                     epoch)
                    if args.data_type == 'cqt':
                        loader = test_loader if args.test_cqt else train_loader
                        plot_pca(loader, fn=f"pca_trans_{epoch}.png")
                else:
                    break
            torch.save(model.state_dict(), model_save_fn)

    # Save the model
    torch.save(model.state_dict(), model_save_fn)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a complex '
                                                 'autoencoder on different '
                                                 'types of data')

    parser.add_argument('run_keyword', type=str, default="experiment1",
                        help='keyword used for output path')
    parser.add_argument('input_files', type=str, default="",
                        help='text file containing a list of text files '
                             'each containing a list of audio files for '
                             'training input')
    parser.add_argument('config', type=str, default="config.ini",
                        help='config file using config_spec.ini as spec')
    parser.add_argument('--refresh-cache', action="store_true", default=False,
                        help='reload and preprocess data')

    args = parser.parse_args()

    config = get_config(args.config, 'config_spec.cfg')
    args = config_to_args(config, args)

    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    out_dir = os.path.join("output", args.run_keyword)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.data_type == 'cqt':
        args.block_size = -1 if args.test_cqt else args.block_size

        files = prepare_audio_inputs(args.input_files)
        data = get_cqts(files, cache_key='train', rebuild=args.rebuild or
                                                          args.refresh_cache,
                        use_nr_samples=args.block_size, sr=args.sr, args=args)
        nr_eval = len(data) // 5
        data_eval = data[-nr_eval:]
        data = data[:-nr_eval]

        train_loader, eval_loader, test_loader = \
            create_data_loaders(data, data_eval, data_eval,
                                length_ngram=args.length_ngram,
                                samples_epoch=args.samples_epoch,
                                batch_size=args.batch_size,
                                transform=args.transform,
                                emph_onset=args.emph_onset)

        in_size = args.n_bins * args.length_ngram
    elif args.data_type == 'mnist':
        in_size = 28 * 28
        data = load_pyc_bz("./data/mnist_rot.pyc.bz")
        np.random.shuffle(data)

        nr_eval = len(data) // 5
        data_eval = data[-nr_eval:]
        data = data[:-nr_eval]
        data = [(x[0], x[1]) for x in data]
        data_eval = [(x[0], x[1]) for x in data_eval]
        data = np.array(data)
        data_eval = np.array(data_eval)
        train_loader, eval_loader = create_data_mnist(data,
                                                      data_eval,
                                                      args.batch_size)
    elif args.data_type == 'signal':
        in_size = args.length_ngram
        train_loader, eval_loader = create_data_sig(in_size,
                                                    refresh_cache=args.rebuild)

    model = Complex(in_size, args.n_bases, dropout=args.dropout,
                    learn_norm=args.learn_norm)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_save_fn = os.path.join(out_dir, "model_complex_auto_"
    f"{args.data_type}.save")

    if args.train_it:
        if args.continue_train:
            try:
                print(f"Searching for state dict file {model_save_fn}")
                model.load_state_dict(torch.load(model_save_fn), strict=False)
            except:
                pass
        model = run_experiment(model, start_epoch=args.start_epoch)
    else:
        try:
            print(f"Searching for state dict file {model_save_fn} ...")
            state_dict = torch.load(model_save_fn)
            model.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print(f"... not found.")
            model = run_experiment(model, start_epoch=args.start_epoch)
