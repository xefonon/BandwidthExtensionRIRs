import h5py
import numpy as np
# from glob import glob
import os
import json
import yaml
import time
import math
import torch
import matplotlib.pyplot as plt
import glob
# from icecream import ic
import re

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)
def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)
def config_from_yaml(yamlFilePath, no_description = True):
    def changer(config):
        for attr, value in vars(config).items():
            setattr(config, attr, value.value)

    with open(yamlFilePath) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
    config = dict2obj(dataMap)
    if no_description:
        changer(config)
    return config

def print_training_stats(epoch, num_epochs, steps_per_epoch, loss, i, start_time = None):
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
    Example:

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
        elif isinstance(v, (np.ndarray, np.generic) ):
            return v[-1]
        else:
            return v

    if start_time is not None:
        total_minutes, total_seconds = time_since(start_time)
        print(f"\r\033[95m[Time Passed {round(total_minutes, 2)} min, {round(total_seconds, 2)} sec]\033[0m"
              f"\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
              f"\033[93m[Batch {i%steps_per_epoch + 1}/{steps_per_epoch}]\033[0m", end = '', flush= False)
        print(", ".join([' {} = {!r}'.format(k, np.round(check_list(v), 4)) for k, v in loss.items()]),  end='', flush= True)

    else:
        print(f"\r\033[92m[Epoch {epoch + 1}/{num_epochs}]\033[0m"
                   f"\033[93m[Batch {i%steps_per_epoch + 1}/{steps_per_epoch}]\033[0m", end = '', flush= True)
        print(", ".join([' {} = {!r}'.format(k, np.round(v, 4)) for k, v in loss.items()]),  end='', flush= True)


def to_db(H_in, norm=False):
    if norm:
        H_in /= np.max(abs(H_in))
    H_db = 20 * np.log10(abs(H_in))
    return H_db

def evaluate_response(Generator,
                      eval_files,
                      input_size =8192,
                      batch_size = 16,
                      plot = False,
                      use_fft = False,
                      zoff = False):
    # eval on real data
    f_ind = np.random.randint(low = 0, high = 20, size = batch_size)
    # f = eval_files[0]
    # gt_responses, eval_responses = read_npy_file(f)
    gt_responses, eval_responses = eval_files
    if not use_fft:
        gt_responses = gt_responses[f_ind, :input_size]
        eval_responses = eval_responses[f_ind, :input_size]
    else:
        gt_responses = gt_responses[f_ind, :]
        eval_responses = eval_responses[f_ind, :]
    # batch of real RIR's
    ground_truths = torch.tensor(gt_responses)
    in_responses = torch.tensor(eval_responses)

    Generator.training = False

    out_response = Generator(in_responses)

    cos_sim = torch.nn.CosineSimilarity(dim = -1)(out_response, ground_truths)
    if plot:
        plt_ind = np.random.choice(len(f_ind))
        ax1 = plot_rir(ground_truths[plt_ind, :], in_responses[plt_ind, :], title= 'Real RIR evaluation',
                       y_pred_lab= 'GAN input', t_intervals = [0, 0.2])

        ax2 = plot_rir(ground_truths[plt_ind, :], out_response[plt_ind, :], title= 'Real RIR evaluation',
                       y_pred_lab= 'GAN output', t_intervals = [0, 0.2])

        return cos_sim, ax1, ax2
    else:
        return cos_sim

def normalise_response(h_in):
    h_in /= np.max(abs(h_in))
    return h_in

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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

class VariableGrid():
    """
    Example Use:

    fig = plt.figure()
    grid = VariableGrid(fig)
    for ii in range(5):
        grid.add()
    test = np.random.randn(5, 10000)
    for ii in range(len(test)):
        plot_rir(test[ii],grid.axes[ii])
    plt.show()
    """
    def __init__(self,fig):
        self.fig = fig
        self.axes = []
        self.gs = None
        self.n = 0

    def update(self):
        if self.n > 0:
            for i,ax in zip(range(self.n), self.axes):
                ax.set_position(self.gs[i-1].get_position(self.fig))
                ax.set_visible(True)

            for j in range(len(self.axes),self.n,-1 ):
                # print(self.n, j)
                self.axes[j-1].set_visible(False)
        else:
            for ax in self.axes:
                ax.set_visible(False)
        self.fig.canvas.draw_idle()


    def add(self, evt=None):
        self.n += 1
        self.gs= GridSpec(self.n,1)
        if self.n > len(self.axes):
            ax = self.fig.add_subplot(self.gs[self.n-1], sharex = self.axes[0] if self.axes else None )

            self.axes.append(ax)
        for ii in range(-1, -len(self.axes), -1):
            # print(ii)
            plt.setp(self.axes[ii].get_xticklabels(), visible=False)
        self.update()

    def sub(self, evt=None):
        self.n = max(self.n-1,0)
        if self.n > 0:
            self.gs= GridSpec(self.n,1)
        self.update()

def plot_rir(rir, ax, fs = 16000, y_labels=None, t_intervals=None):
    # from scipy.spatial.distance import cosine
    if t_intervals is None:
        t_intervals = [.01, .2]
    t = np.linspace(0, rir.shape[-1] / fs, rir.shape[-1])
    t_ind = np.argwhere((t > t_intervals[0]) & (t < t_intervals[1]))[:, 0]
    ax.plot(t[t_ind], rir[t_ind], linewidth=3, color='k')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Normalised SP [Pa]')
    ax.set_ylim([np.min(rir) - .1 * np.max(rir), np.max(rir) + .1 * np.max(rir)])
    ax.grid(linestyle=':', which='both', color='k')
    if y_labels is None:
        y_labels = 'GAN Impulse Response'

    # leg = ax.legend([y_labels])
    # ax.set_title(title)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(2.0)
    return ax

def time_since(since):
    now = time.time()
    sec = now - since
    minutes = math.floor(sec / 60)
    sec -= minutes * 60
    return minutes, sec


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(directory, filepath, obj, remove_below_step = None):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    if remove_below_step is not None:
        print("\nRemoving checkpoints below step ", remove_below_step)
        remove_checkpoint(directory, remove_below_step)

def remove_checkpoint(cp_dir, delete_below_steps = 1000):
    D_filelist = [ f for f in os.listdir(cp_dir) if f.startswith("d") ]
    G_filelist = [ f for f in os.listdir(cp_dir) if f.startswith("g") ]
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
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def plot_generated_samples(samples, fs = 16000):
    n_samples = len(samples)
    fig = plt.figure()
    grid = VariableGrid(fig)
    for ii in range(n_samples):
        grid.add()
    for ii in range(n_samples):
        plot_rir(samples[ii], grid.axes[ii], fs = fs)
    grid.axes[0].set_xlabel('Time [s]')
    grid.axes[int(np.ceil(n_samples/2))].set_ylabel('Pressure')
    return fig