import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import math
import time
from glob import glob
import torch
import os
import re


class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)


def config_from_yaml(yamlFilePath, no_description=True):
    def changer(config):
        for attr, value in vars(config).items():
            try:
                setattr(config, attr, value.value)
            except:
                ValueError("problem with config: {} and value: {}".format(config, value))

    with open(yamlFilePath) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
    config = dict2obj(dataMap)
    if no_description:
        changer(config)
    return config

def find_files(root_dir, query="*.npz", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def print_training_stats(epoch, num_epochs, steps_per_epoch, loss, i, start_time=None):
    """
    Print training stats (logging) when training a model

    Parameters
    ----------
    epoch - Epoch number (int)
    num_epochs - Total epochs to run for (int)
    steps_per_epoch - Total batches or steps to run each epoch for (int)
    loss - A dictionary with the various loss values and metrics to log
    i - the step in epoch 'epoch'
    start_time : time.time() (timestamp) when training is initialised

    -------
    Example usage:

    loss = {'one': 1, 'two': 2, 'three': 3}
    start_time = time.time()
    for ii in range(epochs):
        for jj in range(n_batches):
            time.sleep(0.1)
            print_training_stats(ii, epochs, n_batches, loss, jj, start_time = start_time)

    """

    def check_list(v):
        if isinstance(v, list):
            return v[-1]
        elif isinstance(v, (np.ndarray, np.generic)):
            return v[-1]
        else:
            return v

    if start_time is not None:
        total_minutes, total_seconds = time_since(start_time)
        print(f"\r\033[95m[Time Passed {round(total_minutes, 2)} min, {round(total_seconds, 2)} sec]\033[0m"
              f"\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
              f"\033[93m[Batch {i % steps_per_epoch + 1}/{steps_per_epoch}]\033[0m", end='', flush=False)
        print(", ".join([' {} = {!r}'.format(k, np.round(check_list(v), 4)) for k, v in loss.items()]), end='',
              flush=True)

    else:
        print(f"\r\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
              f"\033[93m[Batch {i % steps_per_epoch + 1}/{steps_per_epoch}]\033[0m", end='', flush=True)
        print(", ".join([' {} = {!r}'.format(k, np.round(v, 4)) for k, v in loss.items()]), end='', flush=True)


def time_since(since):
    now = time.time()
    sec = now - since
    minutes = math.floor(sec / 60)
    sec -= minutes * 60
    return minutes, sec


def to_db(H_in, norm=False):
    if norm:
        H_in /= np.max(abs(H_in))
    H_db = 20 * np.log10(abs(H_in))
    return H_db


def normalise_response(h_in):
    h_in /= np.max(abs(h_in))
    return h_in

def generator_loss_fn(loss_type):
    if loss_type == 'logcosh':
        loss_fn = src.models.SEGAN.auraloss.time.LogCoshLoss()
    elif loss_type == 'SISDR':
        loss_fn =  src.models.SEGAN.auraloss.time.SISDRLoss(eps = 1e-16)
    elif loss_type == 'random_res_STFT':
        loss_fn = src.models.SEGAN.auraloss.freq.RandomResolutionSTFTLoss(max_fft_size = 8192, sample_rate = 16000)
    elif loss_type == 'ESRLoss':
        loss_fn = src.models.SEGAN.auraloss.time.ESRLoss()
    elif loss_type == 'SumDiffSTFT':
        loss_fn = src.models.SEGAN.auraloss.freq.SumAndDifferenceSTFTLoss()
    elif loss_type == 'multiSTFT':
        loss_fn = src.models.SEGAN.auraloss.freq.MultiResolutionSTFTLoss()
    else:
        loss_fn = torch.nn.L1Loss()
    return loss_fn

def plot_frf(y_truth, y_pred, title='', y_pred_lab=None):
    from matplotlib.ticker import FuncFormatter
    def kilos(x, pos):
        'The two args are the value and tick position'
        if x < 10:
            return None
        elif x == 500:
            return '%1.1fk' % (x * 1e-3)
        elif x % 1000 == 0:
            return '%1dk' % (x * 1e-3)
        elif x >= 1000:
            return '%1.1fk' % (x * 1e-3)

    fs = 16000
    # y_truth = y_truth[:8192]
    # y_pred = y_pred[:8192]
    y_pred_np = y_pred.cpu()
    # y_pred_np= y_pred_np[0,:]
    # y_truth= y_truth[0,:]

    freq = np.fft.rfftfreq(len(y_truth), d=1 / fs)
    freq_ind = np.argwhere(freq > 499)[:, 0]
    # fr_truth = normalise_response(np.fft.rfft(y_truth))
    # fr_pred = normalise_response(np.fft.rfft(y_pred))
    fr_truth = np.fft.rfft(y_truth)
    fr_pred = np.fft.rfft(y_pred_np)
    Y_rec = to_db(fr_pred)
    Y_truth = to_db(fr_truth)
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    # ax = plt.gca()
    ax.semilogx(freq[freq_ind], Y_truth[freq_ind], linewidth=3, color='k')
    ax.semilogx(freq[freq_ind], Y_rec[freq_ind], linewidth=1, color='seagreen')
    # ax.semilogx(freq, Y_truth, linewidth=3, color='k')
    # ax.semilogx(freq, Y_rec, linewidth=1, color='seagreen')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Normalised SPL [dB]')
    formatter = FuncFormatter(kilos)

    ax.xaxis.set_minor_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    if y_pred_lab is None:
        y_pred_lab = 'Reconstructed frequency response'

    ax.set_ylim([np.min(Y_truth) - 10, np.max(Y_truth) + 10])
    ax.grid(linestyle=':', which='both', color='k')
    leg = ax.legend(['Synthesised PW frequency response', y_pred_lab])
    ax.set_title(title)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.show()
    return ax


def plot_rir(y_truth, y_pred, title='', y_pred_lab=None, t_intervals=None):
    # from scipy.spatial.distance import cosine
    if t_intervals is None:
        t_intervals = [.01, .2]
    y_pred = y_pred.cpu()
    # y_pred= y_pred[0,:]
    # y_truth= y_truth[0,:]

    y_truth = normalise_response(y_truth.numpy())
    y_pred = normalise_response(y_pred.numpy())
    fs = 16000
    t = np.linspace(0, len(y_truth) / fs, len(y_truth))
    t_ind = np.argwhere((t > t_intervals[0]) & (t < t_intervals[1]))[:, 0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    ax.plot(t[t_ind], y_truth[t_ind], linewidth=3, color='k')
    ax.plot(t[t_ind], y_pred[t_ind], linewidth=1, color='seagreen')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Normalised SP [Pa]')
    ax.set_ylim([np.min(y_truth) - .1 * np.max(y_truth), np.max(y_truth) + .1 * np.max(y_truth)])
    # ic(y_truth.shape)
    # ic(y_pred.shape)
    ax.annotate('Corr. Coeff.: {:.2f}'.format(np.corrcoef(y_truth[:int(.5 * fs)],
                                                          y_pred[:int(.5 * fs)])[0, 1]),
                xy=(.9 * t[t_ind].max(), 1.05 * np.max(y_truth[t_ind])))
    ax.grid(linestyle=':', which='both', color='k')
    if y_pred_lab is None:
        y_pred_lab = 'GAN Impulse Response'

    leg = ax.legend(['True Impulse Response', y_pred_lab])
    ax.set_title(title)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    return ax


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(directory, filepath, obj, remove_below_step=None):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    if remove_below_step is not None:
        print("\nRemoving checkpoints below step ", remove_below_step)
        remove_checkpoint(directory, remove_below_step)


def remove_checkpoint(cp_dir, delete_below_steps=1000):
    D_filelist = [f for f in os.listdir(cp_dir) if f.startswith("d")]
    G_filelist = [f for f in os.listdir(cp_dir) if f.startswith("g")]
    for f in D_filelist:
        prefix, number, extension = re.split(r'(\d+)', f)
        if int(number) < delete_below_steps:
            os.remove(os.path.join(cp_dir, f))
    for f in G_filelist:
        prefix, number, extension = re.split(r'(\d+)', f)
        if int(number) < delete_below_steps:
            os.remove(os.path.join(cp_dir, f))


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
