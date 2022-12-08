import h5py
import numpy as np
# from glob import glob
import os
import json
import yaml
import time
import math
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import matplotlib as mpl
from scipy.interpolate import griddata

def plot_sf(P, x, y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim= None, tex=False):
    N_interp = 1500
    Pvec = P/np.max(abs(P))
    res = np.complex(0, N_interp)
    Xc, Yc = np.mgrid[x.min():x.max():res, y.min():y.max():res]
    points = np.c_[x, y]
    Pmesh = griddata(points, Pvec, (Xc, Yc), method='cubic',  rescale = True)

    cmap = 'coolwarm'
    if f is None:
        f = ''
    # P = P / np.max(abs(P))
    X = Xc.flatten()
    Y = Yc.flatten()
    if tex:
        plt.rc('text', usetex=True)
    # x, y = X, Y
    # clim = (abs(P).min(), abs(P).max())
    dx = 0.5 * X.ptp() / Pmesh.size
    dy = 0.5 * Y.ptp() / Pmesh.size
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(np.real(Pmesh), cmap=cmap, origin='upper',
                   extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    lm1, lm2 = clim
    im.set_clim(lm1, lm2)
    cbar = plt.colorbar(im)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('Normalised SPL [dB]', rotation=270)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        ax.set_title(name + ' - f : {} Hz'.format(f))
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax

def plot_settings():
    width = 6.694

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }

    mpl.rcParams.update(tex_fonts)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    # plt.rcParams["figure.figsize"] = (6.694, 5)
    plt.rcParams['figure.constrained_layout.use'] = True
    return width

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

    freq = np.fft.rfftfreq(len(y_truth), d=1 / fs)
    freq_ind = np.argwhere(freq > 499)[:, 0]
    fr_truth = np.fft.rfft(y_truth)
    fr_pred = np.fft.rfft(y_pred)
    Y_rec = to_db(fr_pred, norm = True)
    Y_truth = to_db(fr_truth, norm = True)
    width = plot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(width, width/4))
    ax.semilogx(freq[freq_ind], Y_truth[freq_ind], linewidth=3, color='k')
    ax.semilogx(freq[freq_ind], Y_rec[freq_ind], linewidth=1, color='seagreen')

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

def plot_rir(y_truth, y_pred, title='', y_pred_lab=None, t_intervals=None):
    if t_intervals is None:
        t_intervals = [.01, .2]

    width = plot_settings()
    # fig, ax = plt.subplots(1, 1, figsize=(width, width/4))

    y_truth = normalise_response(y_truth)
    y_pred = normalise_response(y_pred)
    fs = 16000
    t = np.linspace(0, len(y_truth) / fs, len(y_truth))
    t_ind = np.argwhere((t > t_intervals[0]) & (t < t_intervals[1]))[:, 0]
    fig, ax = plt.subplots(1, 1, figsize=(width, width/4))
    ax.plot(t[t_ind], y_truth[t_ind], linewidth=3, color='k')
    ax.plot(t[t_ind], y_pred[t_ind], linewidth=1, color='seagreen')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Normalised SP [Pa]')
    ax.set_ylim([np.min(y_truth) - .1 * np.max(y_truth), np.max(y_truth) + .1 * np.max(y_truth)])
    ax.annotate('Corr. Coeff.: {:.2f}'.format(np.corrcoef(y_truth[:int(.5 * fs)],
                                                          y_pred[:int(.5 * fs)])[0, 1]),
                xy=(.9 * t[t_ind].max(), 1.05 * np.max(y_truth[t_ind])))
    ax.grid(linestyle=':', which='both', color='k')
    if y_pred_lab is None:
        y_pred_lab = 'GAN Impulse Response'

    leg = ax.legend(['True Impulse Response', y_pred_lab], loc = 'lower right')
    ax.set_title(title)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
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

def find_files(root_dir, query="*.npz", include_root_dir=True):
    import fnmatch
    import os
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

def validation_responses(root_dir, query="*.hdf5"):
    file_name = find_files(root_dir, query = query)
    with h5py.File(file_name[0], "r") as f:
        valid_true = f['ground_truth'][:]
        valid_recon = f['reconstructions'][:]
    f.close()
    return valid_true, valid_recon

def select_responses(grid, index = None, mode = 'interpolated'):
    if index is None:
        if mode == 'interpolated':
            indices = np.argwhere(grid[0]**2 + grid[1]**2 < 0.5)
        else:
            indices = np.argwhere(grid[0]**2 + grid[1]**2 > 0.69)

def normalize(input, norm_ord = np.inf):
    norm = torch.linalg.norm(input.squeeze(0), ord = norm_ord, keepdim = True)
    input = input/norm
    return input.unsqueeze(0)