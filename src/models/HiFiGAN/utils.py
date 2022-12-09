import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm

# matplotlib.use("Agg")
import matplotlib.pylab as plt
import matplotlib as mpl
import fnmatch
import os
import yaml
import numpy as np
import h5py
import re
import json
# from icecream import ic


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

def to_db(H_in, norm=False):
    if norm:
        H_in /= np.max(abs(H_in)[20:-20])
    H_db = 20 * np.log10(abs(H_in))
    return H_db


def evaluate_response(
    Generator,
    eval_files,
    input_size=8192,
    batch_size=16,
    plot=False,
    use_fft=False,
    zoff=False,
):
    # eval on real data
    f_ind = np.random.randint(low=0, high=20, size=batch_size)
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

    cos_sim = torch.nn.CosineSimilarity(dim=-1)(out_response, ground_truths)
    if plot:
        plt_ind = np.random.choice(len(f_ind))
        ax1 = plot_rir(
            ground_truths[plt_ind, :],
            in_responses[plt_ind, :],
            title="Real RIR evaluation",
            y_pred_lab="GAN input",
            t_intervals=[0, 0.2],
        )

        ax2 = plot_rir(
            ground_truths[plt_ind, :],
            out_response[plt_ind, :],
            title="Real RIR evaluation",
            y_pred_lab="GAN output",
            t_intervals=[0, 0.2],
        )

        return cos_sim, ax1, ax2
    else:
        return cos_sim


def normalise_response(h_in):
    h_in /= np.max(abs(h_in))
    return h_in

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
    freq_ind = np.argwhere(freq > 20)[:, 0]
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

    leg = ax.legend(['True Impulse Response', y_pred_lab])
    ax.set_title(title)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    return ax


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


def load_hparam_str(hp_str):
    path = "temp-restore.yaml"
    with open(path, "w") as f:
        f.write(hp_str)
    ret = HParam(path)
    os.remove(path)
    return ret


def load_hparam(filename):
    stream = open(filename, "r")
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = Dotdict(value)
            self[key] = value


class HParam(Dotdict):
    def __init__(self, file):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


def validation_responses(valid_path, extension="*.hdf5"):

    files = find_files(valid_path, extension)
    assert len(files) > 0, "No .hdf5 files found"
    with h5py.File(files[0], "r") as f:
        valid_true = f["ground_truth"]
        valid_recon = f["reconstructions"]
        # indx = np.random.randint(0, len(valid_true))
    return valid_true, valid_recon


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

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

def save_checkpoint(directory, filepath, obj, remove_below_step=None):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    if remove_below_step is not None:
        print("\nRemoving checkpoints below step ", remove_below_step)
        remove_checkpoint(directory, remove_below_step)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, "w") as h5file:
        for key, item in dic.items():
            if isinstance(item, list):
                h5file[key] = np.asarray(item)
            else:
                h5file[key] = item
        h5file.close()
