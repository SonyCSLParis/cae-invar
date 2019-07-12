"""
This file contains a set of useful functions for the implementation of
the music extractor algorithm.

########

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
"""

import pickle as cPickle
import csv
import numpy as np
import os
import pylab as plt
from scipy import spatial
import logging

# import librosa

CSV_ONTIME = 0
CSV_MIDI = 1
CSV_HEIGHT = 2
CSV_DUR = 3
CSV_STAFF = 4


def ensure_dir(dir):
    """Makes sure that the directory dir exists.

    Parameters
    ----------
    dir: str
        Path to the directory to be created if needed.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def plot_matrix(X):
    """Plots matrix X."""
    plt.imshow(X, interpolation="nearest", aspect="auto")
    plt.show()


def read_cPickle(file):
    """Reads the cPickle file and returns its contents.

    @param file string: Path to the cPickle file.
    @return x Object: cPickle contents.
    """
    f = open(file, "r")
    x = cPickle.load(f)
    f.close()
    return x


def write_cPickle(file, data):
    """Write data into a cPickle file.

    @param file string: Path to the new cPickle file.
    @param data object: Data to be stored.
    """
    f = open(file, "w")
    cPickle.dump(data, f, protocol=1)
    f.close()


def compute_ssm(X, h, dist="euclidean"):
    """Compute a Self Similarity Matrix, normalized from 0 to 1."""
    L = int(1. / h)
    if L % 2 == 0:
        L += 1
    X = median_filter(X, L=L)
    S = spatial.distance.pdist(X, dist)
    S = spatial.distance.squareform(S)
    S /= S.max()
    S = 1 - S
    return S


def compute_key_inv_ssm(X, h, dist="euclidean"):
    """Computes the self similarity matrix that is key invariant from
        chromagram X."""
    P = X.shape[1]
    L = int(1. / h)
    if L % 2 == 0:
        L += 1
    if L <= 1:
        L = 9
    X = median_filter(X, L=L)
    N = X.shape[0]
    SS = np.zeros((P, N, N))
    dist = "euclidean"
    for i in range(P):
        SS[i] = spatial.distance.cdist(X, np.roll(X, i), dist)

    # import ismir
    # K = SS[0] / SS[0].max()
    # ismir.plot_ssm(1 - K)

    # Get key-ivariant ssm:
    S = np.min(SS, axis=0)

    """
    S = spatial.distance.pdist(X, metric=dist)
    S = spatial.distance.squareform(S)
    """

    # Normalize
    S /= S.max()
    S = 1 - S

    return S


def chroma_to_tonnetz(C):
    """Transforms chromagram to Tonnetz (Harte, Sandler, 2006)."""
    N = C.shape[0]
    T = np.zeros((N, 6))

    r1 = 1  # Fifths
    r2 = 1  # Minor
    r3 = 0.5  # Major

    # Generate Transformation matrix
    phi = np.zeros((6, 12))
    for i in range(6):
        for j in range(12):
            if i % 2 == 0:
                fun = np.sin
            else:
                fun = np.cos

            if i < 2:
                phi[i, j] = r1 * fun(j * 7 * np.pi / 6.)
            elif i >= 2 and i < 4:
                phi[i, j] = r2 * fun(j * 3 * np.pi / 2.)
            else:
                phi[i, j] = r3 * fun(j * 2 * np.pi / 3.)

    # Do the transform to tonnetz
    for i in range(N):
        for d in range(6):
            denom = float(C[i, :].sum())
            if denom == 0:
                T[i, d] = 0
            else:
                T[i, d] = 1 / denom * (phi[d, :] * C[i, :]).sum()

    return T


def get_smaller_dur_csv(score, thres=False):
    """Gets the smaller duration of a csv formatted score."""
    min_dur = np.min(score[:, CSV_DUR])
    if thres:
        if min_dur < 0.25:
            min_dur = .25
    return min_dur


def get_total_dur_csv(score):
    """Computes the total duration of a csv formatted score."""
    # Get indeces for last note
    max_onsets = np.argwhere(score[:, CSV_ONTIME] ==
                             np.max(score[:, CSV_ONTIME]))

    # Get max dur for last note
    max_dur = np.max(score[max_onsets, CSV_DUR])

    # Get anacrusa (pick up)
    min_onset = get_offset(score)
    if min_onset > 0:
        min_onset = 0

    # Compute total time
    total_dur = score[max_onsets[0], CSV_ONTIME] + max_dur + np.abs(min_onset)

    return total_dur


def get_number_of_staves(score):
    """Returns the number of staves for the csv formatted score."""
    return int(np.max(score[:, CSV_STAFF])) + 1


def get_offset(score):
    """Returns the offset (pick up measure), if any, from the score."""
    return np.min(score[:, CSV_ONTIME])


def midi_to_chroma(pitch):
    """Given a midi pitch (e.g. 60 == C), returns its corresponding
        chroma class value. A == 0, A# == 1, ..., G# == 11 """
    return ((pitch % 12) + 3) % 12


def csv_to_chromagram(score):
    """Obtains a chromagram representation from a csv score."""
    # get smaller duration
    h = get_smaller_dur_csv(score, thres=True)

    # Get the total duration
    total_dur = get_total_dur_csv(score)

    # Init the Chromagra
    N = np.ceil(total_dur / float(h))
    C = np.zeros((N, 12))

    # offset
    offset = np.abs(int(get_offset(score) / float(h)))

    # Compute the chromagram
    for row in score:
        pitch = midi_to_chroma(row[CSV_MIDI])
        start = int(row[CSV_ONTIME] / float(h)) + offset
        end = start + int(row[CSV_DUR] / float(h))
        C[start:end, pitch] = 1

    return C, h


def median_filter(X, L=9):
    """Applies a median filter of size L to the matrix of row
        observations X."""
    Y = np.ones(X.shape) * X.min()
    Lh = (L - 1) / 2
    for i in np.arange(Lh, X.shape[0] - Lh):
        Y[i, :] = np.median(X[i - Lh:i + Lh, :], axis=0)
    return Y


def mean_filter(X, L=9):
    """Applies a mean filter of size L to the matrix of row observations X."""
    Y = np.ones(X.shape) * X.min()
    Lh = (L - 1) / 2
    for i in np.arange(Lh, X.shape[0] - Lh):
        Y[i, :] = np.mean(X[i - Lh:i + Lh, :], axis=0)
    return Y


def is_square(X, start_i, start_j, M, th):
    """Checks whether the block of the ssm defined with start_i, start_j
        contains "squared" information or not."""
    try:
        subX = X[start_i:start_i + M, start_j:start_j + M]
        rho = 1
        if subX.trace(offset=rho) >= (M - rho * 2) * th or \
                subX.trace(offset=-rho) >= (M - rho * 2) * th:
            return True
        else:
            return False
    except:
        return False


def split_patterns(patterns, max_diff, min_dur):
    """Splits the patterns in case they are included one inside the other."""
    s_patterns = []

    N = len(patterns)
    splitted = np.zeros(N)
    for i in range(N):
        o1 = patterns[i][0]
        for j in range(N):
            if i == j:
                continue
            if splitted[j]:
                continue
            o2 = patterns[j][0]

            # Check if we have to split
            if o1[0] > o2[0] and o1[1] < o2[1] and \
                    ((o2[1] - o1[1]) - (o1[1] - o1[0]) > max_diff):
                new_p = []

                # Add original pattern
                for p in patterns[i]:
                    new_p.append(p)

                # Add splitted pattern
                for k, p in enumerate(patterns[j]):
                    if k == 0:
                        continue  # Do not add the diagonal repetition
                    start_j = p[2] + (o1[0] - o2[0])
                    end_j = p[3] - (o2[1] - o1[1])
                    new_p.append([o1[0], o1[1], start_j, end_j])

                # Add new pattern to the splitted ones
                s_patterns.append(new_p)

                # Create new pattern from the first part if needed
                if o1[0] - o2[0] > min_dur:
                    first_new_p = []
                    for p in patterns[j]:
                        end_j = p[2] + (o1[0] - p[0])
                        first_new_p.append([p[0], o1[0], p[2], end_j])
                    s_patterns.append(first_new_p)

                # Create new pattern from the last part if needed
                if o2[1] - o1[1] > min_dur:
                    last_new_p = []
                    for p in patterns[j]:
                        start_j = p[3] - (p[1] - o1[1])
                        last_new_p.append([o1[1], p[1], start_j, p[3]])
                    s_patterns.append(last_new_p)

                # Marked a splitted
                splitted[i] = 1
                splitted[j] = 1

    # Add the rest of non-splitted patterns
    for i in range(N):
        if splitted[i] == 0:
            new_p = []
            for p in patterns[i]:
                new_p.append(p)
            s_patterns.append(new_p)

    return s_patterns


def save_results(csv_patterns, outfile="results.txt"):
    """Saves the results into the output file, following the MIREX format."""
    f = open(outfile, "w")
    P = 1
    for pattern in csv_patterns:
        f.write("pattern%d\n" % P)
        O = 1
        for occ in pattern:
            f.write("occurrence%d\n" % O)
            for row in occ:
                f.write("%f, %f\n" % (row[0], row[1]))
            O += 1
        P += 1
    f.close()


def save_results_raw(csv_patterns, outfile="results.txt"):
    """Saves the raw results into the output file"""
    f = open(outfile, "w")
    P = 1
    for pattern in csv_patterns:
        f.write("pattern%d\n" % P)
        for occ in pattern[1:]:
            to_write = str(np.array(occ)[:2])[1:-1] + "->" + str(np.array(
                occ)[2:])[1:-1] + "\n"
            f.write(to_write)
        P += 1
    f.close()
    print(f"Patterns written to {outfile}.")


def compute_segment_score_omega(X, start_i, start_j, min_dur, th, rho):
    """Computes the score for a given segment. The score is computed by
        taking a look at the trace of the block of the ssm."""
    # Compute initial score
    M = 1
    final_score = 0
    while start_i + M < X.shape[0] and start_j + M < X.shape[0]:
        patch = X[start_i:start_i + M, start_j:start_j + M]

        cons_n_prev = 10

        if M == 1:
            weighted = np.eye(1, 1)
        else:
            weight = np.concatenate((np.zeros(np.maximum(0, M - cons_n_prev)),
                                     np.arange(np.minimum(M, cons_n_prev)) + 1))
            weighted = (weight * np.eye(M, M))

        score = 0
        for omega in np.arange(-rho + 1, rho):
            score += (patch * weighted / np.sum(weighted)).trace(offset=omega)
        if score <= th:
            break

        M += 1
        final_score = score

    if M < min_dur:
        final_score = 0

    return final_score, M


def compute_segment_score(X, start_i, start_j, min_dur, th):
    """Computes the score for a given segment. The score is computed by
        taking a look at the trace of the block of the ssm."""
    # Compute initial score
    M = min_dur
    final_score = 0
    while start_i + M < X.shape[0] and start_j + M < X.shape[0]:
        patch = X[start_i:start_i + M, start_j:start_j + M]
        score = 0
        score += patch.trace(offset=0)
        score += patch.trace(offset=1)
        score += patch.trace(offset=-1)
        score -= patch.trace(offset=2)
        score -= patch.trace(offset=-2)
        # score -= patch.trace(offset=3)
        # score -= patch.trace(offset=-3)
        score /= (patch.shape[0] + 2 * (patch.shape[0] - 1))

        print(score)

        break
        M += 1
        if score <= th:
            break
        final_score = score

    return final_score, M - 1


def find_segments(X, min_dur, th=0.95, rho=2):
    """Finds the segment inside the self similarity matrix X."""
    N = X.shape[0]
    segments = []
    counter = 0
    for i in range(N - min_dur):
        for j in range(i + 1, N - min_dur):
            # print i, j, min_dur
            # Compute score and assign
            score, M = compute_segment_score_omega(X, i, j, min_dur, th, rho)
            if score > th:  # and not is_square(X, i, j, M, 0.97):
                print(score, i, j, M)
                #                 plt.imshow(X[i:i+M, j:j+M], interpolation="nearest")
                #                 plt.show()
                for k in range(-3, 4):
                    X[i:i + M, j:j + M] *= (1 - np.eye(M, M, k=k))
                segments.append([i, i + M, j, j + M])

            # Counter stuff
            counter += 1
            if counter % (10 * N) == 0:
                print("\t------ %.2f %%" % \
                      (counter / float(N * (N - 1) / 2.) * 100))
    return segments


def read_csv(csv_file):
    """Reads a csv into a numpy array."""
    f = open(csv_file, "r")
    csvscore = csv.reader(f, delimiter=",")
    score = []
    for row in csvscore:
        score.append([float(row[CSV_ONTIME]), float(row[CSV_MIDI]),
                      float(row[CSV_HEIGHT]), float(row[CSV_DUR]),
                      float(row[CSV_STAFF])])
    score = np.asarray(score)
    f.close()
    return score


def read_wav(wav_file):
    """Reads the wav file and downsamples to 11025 Hz."""
    assert os.path.isfile(wav_file), \
        'ERROR: wivefile file %s does not exist' % wav_file

    x, fs = librosa.core.load(wav_file, sr=11025)
    # if len(x.shape) >= 2:
    #     x = x[:, 0]  # Make mono

    assert fs == 11025, "ERROR: File %s is not sampled at 11025 Hz" % wav_file

    return x, fs


def compute_spectrogram(x, wlen, fs):
    """Computes a spectrogram."""
    N = int(fs * wlen)
    nstep = N / 2  # Hop size of 0.5 * N
    nwin = N

    logging.info("Spectrogram: sample rate: %d, hop size: %d, FFT size: %d" %
                 (fs, nstep, N))

    window = np.blackman(nwin)
    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), N / 2))
    x_down = np.zeros(len(nn))
    for i, n in enumerate(nn):
        xseg = x[n - nwin:n]
        x_down[i] = np.mean(xseg)
        z = np.fft.fft(window * xseg, N)
        X[i, :] = np.abs(z[:N / 2])

    return X, N


def freq2bin(f, N, fs):
    return int(f * N / fs)


def compute_CQT_filters(N, fs):
    """Computes the CQT filters."""
    O = 4
    min_f = 55
    filters = np.zeros((12, N / 2))
    for p in range(12):
        for octave in range(O):
            start_bin = freq2bin(min_f * 2 ** (octave + (p - 1) / 12.), N, fs)
            center_bin = freq2bin(min_f * 2 ** (octave + p / 12.), N, fs)
            M = (center_bin - start_bin) * 2 + 1
            filt = np.blackman(M)
            filt /= filt.sum()
            filters[p, start_bin:start_bin + M] = filt

    # plot_matrix(filters)

    return filters


def compute_audio_chromagram(wav_file, h):
    """Computes the PCP."""

    # Read the wav file
    x, fs = read_wav(wav_file)

    # Compute the spectrogram
    X, N = compute_spectrogram(x, h, fs)

    filters = compute_CQT_filters(N, fs)
    C = np.dot(X, filters.T)
    for i in range(C.shape[0]):
        # Normalize (if greater than a certain energy)
        if C[i, :].max() >= 1:
            C[i, :] /= C[i, :].max()

    # Normalize
    C /= C.max()

    # plot_matrix(C.T)

    return C


def sonify_patterns(wav_file, patterns, h, out_dir="sonify"):
    """Sonifies the patterns."""
    # Read the wav file
    x, fs = read_wav(wav_file)

    # Make sure that directory exists
    ensure_dir(out_dir)

    for i, p in enumerate(patterns):
        for j, occ in enumerate(p):
            hop = int(h * fs) / 2.
            start = int(occ[2] * hop)
            end = int(occ[3] * hop)
            audio_pattern = x[start: end]
            file_name = os.path.join(out_dir,
                                     "pattern%d_occ%d_%.1f-%.1f.wav" %
                                     (
                                     i, j, start / float(fs), end / float(fs)))
            librosa.output.write_wav(file_name, audio_pattern, fs)
