"""
Created on April 13, 2018
Edited on July 05, 2019

@author: Stefan Lattner & Maarten Grachten

Sony CSL Paris, France
Institute for Computational Perception, Johannes Kepler University, Linz
Austrian Research Institute for Artificial Intelligence, Vienna

"""
import numpy as np
import librosa
import torch.utils.data as data
import torch
import logging
import PIL

from scipy.signal import get_window
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose, \
        CenterCrop

from complex_auto.util import to_numpy, cached

LOGGER = logging.getLogger(__name__)


def standardize_(ngram):
    ngram = ngram - ngram.mean()
    std = ngram.std()
    if std > 1e-8:
        ngram = .1 * ngram / std
    return ngram


class Data(object):
    def __init__(self, data_x, data_y, standardize=False):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return [standardize_(torch.FloatTensor(self.data_x[index])),
                standardize_(torch.FloatTensor(self.data_y[index])),
                -1, -1, -1]

    def __len__(self):
        return len(self.data_x)


class DataSampler(object):
    def __init__(self, data_x, length_ngram, samples_epoch, standard=True,
                 shifts=[24, 24], scales=[1., 0], shuffle=True,
                 transform=(0, 1, 2), emph_onset=0, random_pairs=False):
        """
        Returns random ngrams from data, can shift and scale data in two
        dimensions

        :param data_x: data (2d)
        :param length_ngram: length of sampled ngrams
        :param samples_epoch: number of samples per epoch
        :param standard: if instances should be standardized
        :param shifts: 2-tuple, maximal random shifts in two dimensions
        :param scales: 2-tuple, maximal random scaling in two dimensions
        :param shuffle: instances are returned in random order
        :param transform: iterable; which transforms should be applied.
                            pitch_shift (0), time shift (1), tempo-change (2)
        :param emph_onset: onsets are emphasized
        :param random_pairs: a pair is sampled using two random (unrelated)
        instances
        """
        self.data_x = data_x
        self.length_ngram = length_ngram
        self.samples_epoch = samples_epoch
        self.standard = standard
        self.max_x = shifts[0]
        self.max_y = shifts[1]
        self.scale_x = scales[0]
        self.scale_y = scales[1]
        self.shuffle = shuffle
        self.transform = transform
        self.emph_onset = emph_onset
        self.random_pairs = random_pairs
        self.check_lengths()

    def check_lengths(self):
        delete = []
        for i, song in enumerate(self.data_x):
            max_ = song.shape[1] - self.length_ngram - self.max_x
            if not self.max_x < max_:
                print(f"Warning: Song number {i} is too short to be used "
                      f"with ngram length {self.length_ngram} and maximal "
                      f"time shift of {self.max_x} (will be ignored)!")
                delete.append(i)
        self.data_x = [i for j, i in enumerate(self.data_x) if j not in
                       delete]

    def __len__(self):
        if not self.shuffle:
            return self.get_ngram_count()
        return self.samples_epoch

    def __getitem__(self, index):
        # Transform: pitch_shift (0), time shift (1), tempo-change (2)
        if self.transform is None:
            # random transform
            transform = np.random.randint(0, 3)
        else:
            transform = np.random.choice(self.transform)

        if self.random_pairs:
            # song_id, start, end = self.get_random_ngram()
            # ngram = self.data_x[song_id][:, start:end].copy()
            # song_id, start, end = self.get_random_ngram()
            # ngram_trans = self.data_x[song_id][:, start:end].copy()
            if np.random.randint(2) == 0:
                [ngram, ngram_trans], song_id = self.get_pairs_same_song()
                label = -1
                transform = -1 # skips transformation codes
            else:
                song_id, start, end = self.get_ngram_by_idx(index)
                ngram = self.data_x[song_id][:, start:end].copy()
        elif self.shuffle:
            song_id, start, end = self.get_random_ngram()
            ngram = self.data_x[song_id][:, start:end].copy()
        else:
            song_id, start, end = self.get_ngram_by_idx(index)
            ngram = self.data_x[song_id][:, start:end].copy()

        # Normalization needed for PIL image processing (scale)
        ngram -= ngram.min()
        if ngram.max() > 1e-6:
            ngram /= ngram.max()

        assert ngram.shape[1] != 0, f"{start}, {end}," \
                                    f"{self.data_x[song_id].shape[1]}, " \
                                    f"{self.max_x}"

        if transform == 1:
            if self.max_x == 0:
                shiftx = 0
            else:
                shiftx = np.random.randint(-self.max_x, self.max_x)

            ngram_trans = self.trans_time_shift(end, song_id, start,
                                                shiftx)
            label = "shiftx" + str(shiftx)

        if transform == 0:
            if self.max_x == 0:
                shifty = 0
            else:
                shifty = np.random.randint(-self.max_y, self.max_y)
            ngram_trans = self.trans_pitch_shift(ngram, shifty)
            label = "shifty" + str(shifty)

        if transform == 2:
            scale_x = 1 + self.scale_x * np.random.rand()
            ngram, ngram_trans, minus = self.trans_speed_change(ngram, scale_x)
            label = scale_x if not minus else -scale_x
            label = "scale" + str(label)
            ngram = to_numpy(ngram)
            ngram_trans = to_numpy(ngram_trans)

        ngram_onset = np.diff(np.concatenate((ngram[:, 0:1], ngram), axis=1),
                                                                     axis=1)
        ngram_trans_onset = np.diff(np.concatenate((ngram_trans[:, 0:1],
                                              ngram_trans), axis=1), axis=1)
        ngram_onset[ngram_onset < 0] = 0
        ngram_trans_onset[ngram_trans_onset < 0] = 0

        ngram = ngram + ngram_onset * self.emph_onset
        ngram_trans = ngram_trans + ngram_trans_onset * self.emph_onset

        if self.standard:
            ngram = self.standardize(ngram)
            ngram_trans = self.standardize(ngram_trans)

        ngram = torch.FloatTensor(ngram).view(-1)
        ngram_trans = torch.FloatTensor(ngram_trans).view(-1)

        return ngram+1e-8, ngram_trans+1e-8, transform, song_id, label

    def get_ngram_count(self):
        count = 0
        count_data = len(self.data_x)
        for i in range(count_data):
            len_data = self.data_x[i].shape[1]
            startmin = 2 * self.max_x
            startmax = len_data - self.length_ngram - 2 * self.max_x
            count += startmax - startmin
        return count

    def get_ngram_by_idx(self, index):
        count = 0
        count_data = len(self.data_x)
        for i in range(count_data):
            len_data = self.data_x[i].shape[1]
            startmin = 2 * self.max_x
            startmax = len_data - self.length_ngram - 2 * self.max_x
            if index >= count and index + startmin < count + startmax:
                song_id = i
                start = index - count + startmin
                break
            count += startmax - startmin

        end = start + self.length_ngram
        return song_id, start, end

    def get_random_ngram(self):
        count_data = len(self.data_x)
        song_id = np.random.randint(0, count_data)
        len_data = self.data_x[song_id].shape[1]
        start = np.random.randint(self.max_x,
                                  len_data - self.length_ngram - self.max_x)
        end = start + self.length_ngram
        return song_id, start, end

    def get_pairs_same_song(self):
        count_data = len(self.data_x)
        song_id = np.random.randint(0, count_data)
        len_data = self.data_x[song_id].shape[1]
        pairs = []
        for i in range(2):
            start = np.random.randint(2 * self.max_x,
                                      len_data - self.length_ngram - 2 * self.max_x)
            end = start + self.length_ngram
            ngram = self.data_x[song_id][:, start:end].copy()
            pairs.append(ngram)

        return pairs, song_id

    def trans_speed_change(self, ngram, scale_x):
        size1 = ngram.shape[1]
        size0 = ngram.shape[0]
        new_size_t_x = int(scale_x * size1)
        new_size_t_y = ngram.shape[0]
        transform_out = Compose([
            ToPILImage(),
            Resize((new_size_t_y, new_size_t_x),
                   interpolation=PIL.Image.NEAREST),
            CenterCrop((size0, size1)),
            ToTensor()
        ])
        ngram_trans = transform_out(torch.FloatTensor(ngram).unsqueeze(0))

        minus = False
        if np.random.randint(0, 2) == 1:
            ngram_ = ngram
            ngram = ngram_trans
            ngram_trans = ngram_
            minus = True

        return ngram, ngram_trans, minus

    def trans_pitch_shift(self, ngram, shifty):
        return to_numpy(self.transp0(torch.FloatTensor(ngram), shifty))

    def trans_time_shift(self, end, song_id, start, shiftx):
        return self.data_x[song_id][:, start + shiftx:end + shiftx]

    def standardize(self, ngram):
        ngram = ngram - ngram.mean()
        std = ngram.std()
        ngram = .1 * ngram / (std + 1e-8)
        return ngram

    def transp0(self, x, shift):
        """
        Transposes axis 0 (zero-based) of x by [shift] steps.
        Missing information is padded with zeros.
        :param x: the array to transpose
        :param shift: the transposition distance
        :return: x transposed
        """
        if shift == 0:
            return x

        pad = torch.zeros(abs(shift), x.size(1))

        if shift < 0:
            return torch.cat([pad, x[:-abs(shift), :]], dim=0)
        return torch.cat([x[abs(shift):, :], pad], dim=0)

    def transp1(self, x, shift):
        """
        Transposes axis 1 (zero-based) of x by [shift] steps.
        Missing information is padded with zeros.
        :param x: the array to transpose
        :param shift: the transposition distance
        :return: x transposed
        """
        if shift == 0:
            return x

        pad = torch.zeros(x.size(1), abs(shift))

        if shift < 0:
            return torch.cat([pad, x[:, :-abs(shift)]], dim=1)
        return torch.cat([x[:, abs(shift):], pad], dim=1)


class Signal(data.Dataset):

    def __init__(self, filelist, sr="22050", trg_shift=0, block_size=1024,
                 refresh_cache=False, cache_fn="signal_cache.pyc.bz",
                 allow_diff_shapes=False, padded=False, random_shift=0,
                 samples_epoch=1000, window='hann'):
        """
        Constructor for 1D signal dataset

        :param filelist:        list of audio file names (str)
        :param sr:              desired sample rate
        :param trg_shift:       target == input shifted by [-trg_shift] steps,
                                blocks are shortened accordingly
        :param block_size:      length of one instance in a batch
        :param refresh_cache:   when True recalculate and save to cache file
                                when False loads from cache file when available
        :param cache_fn:        filename of cache file

        """
        self.trg_shift = trg_shift
        self.block_size = block_size
        self.sr = sr
        self.allow_diff_shapes = allow_diff_shapes
        self.padded = padded
        self.random_shift = random_shift
        self.window = window
        self.samples_epoch = samples_epoch
        self.signals = cached(cache_fn, self.load_files, (filelist,),
                              refresh_cache=refresh_cache)

    def __getitem__(self, index):
        rand_inst = np.random.randint(len(self.signals))

        if self.random_shift > 0:
            shift = np.random.randint(-self.random_shift, self.random_shift)
        else:
            shift = self.trg_shift

        rand_pos = np.random.randint(abs(shift),
                                     len(self.signals[rand_inst]) -
                                     abs(shift) - self.block_size)

        w = get_window(self.window, self.block_size)
        x = self.signals[rand_inst][rand_pos:rand_pos+self.block_size]
        y = self.signals[rand_inst][rand_pos+shift:
                                    rand_pos+shift+self.block_size, :]
        x = torch.FloatTensor(x.squeeze() * w)
        y = torch.FloatTensor(y.squeeze() * w)

        x = self.standardize(x)
        y = self.standardize(y)

        return x, y, -1, -1, -1

    def standardize(self, signal):
        ngram = signal - signal.mean()
        std = ngram.std()
        if std > 1e-6:
            ngram = ngram / std
        else: # prevent empty input
            ngram = ngram + 1e-8
        return ngram

    def __len__(self):
        return self.samples_epoch

    def load_files(self, filelist):
        data_all = []
        for file in filelist:
            file = file.strip('\n')
            print(f"loading file {file}")
            signal = librosa.load(file)[0][:, None]
            data_all.append(signal)

        if len(data_all) == 0:
            LOGGER.warning("No data added to Signal Dataset!")

        return data_all
