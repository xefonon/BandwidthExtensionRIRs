import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.PyOctaveBand import PyOctaveBand
import h5py
from src.validation_responses.evaluation_metrics import get_eval_metrics
from scipy import stats
import pathlib
from palettable.scientific.sequential import Nuuk_16
import os

n_colors = 4
colorspace = np.linspace(0.0,.8, n_colors)
# colorspace = np.linspace(0.0,1., n_colors)
colors = Nuuk_16.mpl_colormap(colorspace)
figdir = pathlib.Path('FiguresPaper')
figdir.mkdir(parents=True, exist_ok=True)

csgmfilepath = pathlib.Path("..", "models", "CSGM", "inference_data")
hififilepath = pathlib.Path("..", "models", "HiFiGAN", "generated_files")
seganfilepath = pathlib.Path("..", "models", "SEGAN", "generated_files")
npzfilecsgm = os.path.join(str(csgmfilepath), 'inference_data.npz')
npzfilesegan = os.path.join(str(seganfilepath), 'generator_inference_file.npz')
npzfilehifi = os.path.join(str(hififilepath), 'generator_inference_file.npz')
npzfilehifiproc = os.path.join(str(hififilepath), 'generator_inference_processed.npz')
npzfilecsgmproc = os.path.join(str(csgmfilepath), 'generator_inference_processed.npz')
npzfileseganproc = os.path.join(str(seganfilepath), 'generator_inference_processed.npz')


def get_figsize(columnwidth = 246., wf=0.5, hf=(5. ** 0.5 - 1.0) / 2.0, ):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * hf  # height in inches
    return [fig_width, fig_height]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def normalise_response(h_in):
    h_in /= np.max(abs(h_in))
    return h_in


def to_db(H_in, norm=False):
    if norm:
        H_in /= np.max(abs(H_in[50:]))
    H_db = 20 * np.log10(abs(H_in))
    return H_db

def plot_frf(y_truth, y_pred, label=None, color = None, ax = None, normalize = True):
    from matplotlib.ticker import FuncFormatter
    def kilos(x, pos):
        'The two args are the value and tick position'
        if x < 10:
            return None
        elif x == 100:
            return '%1.1fk' % (x * 1e-3)
        # elif x == 500:
        #     return '%1.1fk' % (x * 1e-3)
        elif x == 1000:
            return '%1dk' % (x * 1e-3)
        # elif x >= 1000:
        #     return '%1.1fk' % (x * 1e-3)

    fs = 16000
    if color is None:
        color = colors[0]
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = 'prediction'
    freq = np.fft.rfftfreq(len(y_truth), d=1 / fs)
    freq_ind = np.argwhere(freq > 20)[:, 0]
    fr_truth = np.fft.rfft(y_truth)
    fr_pred = np.fft.rfft(y_pred)
    Y_rec = to_db(fr_pred, norm=normalize)
    Y_truth = to_db(fr_truth, norm=normalize)
    ax.semilogx(freq[freq_ind], Y_truth[freq_ind], linewidth=1, color='k', label = 'Ground truth')
    ax.semilogx(freq[freq_ind], Y_rec[freq_ind], linewidth=0.5, color=color, label = label)
    formatter = FuncFormatter(kilos)
    ax.set_xlim(freq[freq_ind][0], fs/2 + 1)

    ax.xaxis.set_minor_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_yticks([-60, -30, 0])

    ax.set_ylim([-60, 5])
    ax.grid(linestyle=':', which='both', color='k')
    return ax


def plot_rir(y_truth, y_pred, label= None, t_intervals=None, color = None, ax = None):
    if t_intervals is None:
        t_intervals = [.01, .2]
    # fig, ax = plt.subplots(1, 1, figsize=(width, width/4))
    if color is None:
        color = colors[0]
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = 'prediction'

    y_truth = normalise_response(y_truth)
    y_pred = normalise_response(y_pred)
    fs = 16000
    t = np.linspace(0, len(y_truth) / fs, len(y_truth))
    t_ind = np.argwhere((t > t_intervals[0]) & (t < t_intervals[1]))[:, 0]
    ax.plot(t[t_ind], y_truth[t_ind], linewidth=1, color='k', label = 'Ground truth')
    ax.plot(t[t_ind], y_pred[t_ind], linewidth=0.5, color=color, label = label)
    ax.set_ylim([np.min(y_truth) - .1 * np.max(y_truth), np.max(y_truth) + .15 * np.max(y_truth)])
    ax.set_ylim([np.min(y_truth) - .1 * np.max(y_truth), 1.])
    ax.set_xlim([t_intervals[0] - 0.5*t_intervals[0], t_intervals[1]])
    # ax.annotate('Corr. Coeff.: {:.2f}'.format(np.corrcoef(y_truth[:int(.5 * fs)],
    #                                                       y_pred[:int(.5 * fs)])[0, 1]),
    #             xy=(.9 * t[t_ind].max(), 1.05 * np.max(y_truth[t_ind])))
    ax.grid(linestyle=':', which='both', color='k')

    return ax


def calc_MAC(A, B):
    "A, B shape should be frquencies x locations"
    nf = A.shape[0]
    MAC = np.empty((nf), dtype=complex)
    for i in range(nf):
        MAC[i] = np.abs(np.conj(A[i].T) @ B[i]) ** 2 / (
                (np.conj(A[i].T) @ A[i]) * np.conj(B[i].T) @ B[i]
        )
    return MAC


def plot_settings(double_width = False):
    if double_width:
        factor = 2
        tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 11,
            "font.size": 11,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10
        }
    else:
        factor = 1
        tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10
            # "xlabel.fontsize": 10
        }
        tex_fonts = {
            # Use LaTeX to write all text
            # "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 13,
            "font.size": 13,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10
            # "xlabel.fontsize": 10
        }
    fig_width_pt = factor*246.0
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]


    mpl.rcParams.update(tex_fonts)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    # plt.rcParams["figure.figsize"] = (6.694, 5)
    plt.rcParams['figure.constrained_layout.use'] = True
    return fig_size


def config_metrics(h5path):
    """
    For single h5 files
    Parameters
    ----------
    h5path - path to single h5py file

    Returns
    -------

    """

    metrics = {}
    with h5py.File(h5path, 'r') as file:
        # f.visititems(visitor_func)
        for name in file.keys():
            if file[name].ndim < 1:
                metrics[name] = file[name][()]
            else:
                metrics[name] = file[name][:]
        file.close()
    return metrics


def plot_3rdoctave_band_data(freq, data, ylabel="Error [dB]", ax=None, switchcolors=False):
    # Show octave spectrum
    if ax is None:
        ax = plt.gca()
    # if not switchcolors:
    #     colors = ['orangered', 'mediumorchid', 'violet', 'deeppink']
    # else:
    #     colors = ['coral', 'mediumslateblue', 'b', 'steelblue']
    linestyles = ['solid', 'dashed', (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    for ii, key in enumerate(data.keys()):
        ax.semilogx(freq, data[key], color=colors[ii], linestyle=linestyles[ii],
                    linewidth=2, label=key)
    ax.grid(which='major', linestyle=':')
    # ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(ylabel)
    ticks = PyOctaveBand._thirdoctave()
    ax.set_xticks(ticks)
    octave_ticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    octave_tick_labels = ['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k']
    tick_labels = []
    for ii, tick in enumerate(ticks):
        if tick in octave_ticks:
            ind = np.argwhere(np.array(octave_ticks) == tick)
            tick_labels.append(octave_tick_labels[np.squeeze(ind)])
        else:
            tick_labels.append('')

    ax.set_xticklabels(tick_labels)
    ax.minorticks_off()
    ax.legend()
    return ax


def plot_freq_position_data(r, data, ylabel="Error [dB]", ax=None, switchcolors=False):
    # Show octave spectrum
    if ax is None:
        ax = plt.gca()
    # if not switchcolors:
    #     colors = ['orangered', 'mediumorchid', 'violet', 'deeppink']
    # else:
    #     colors = ['coral', 'mediumslateblue', 'b', 'steelblue']
    # linestyles = ['solid', 'dashed', (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    linestyles = ['solid', 'solid','solid','solid']
    for ii, key in enumerate(data.keys()):
        ax.plot(data[key], color=colors[ii], linestyle=linestyles[ii],
                linewidth=2, label=key)
    ax.grid(which='major', linestyle=':')
    # ax.grid(which='minor', linestyle=':')
    # ax.set_xlabel(r'Distance from array center [m]')
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(0, len(r)))
    ticklabs = [str(round(rr, 2)) for rr in r]
    for ii in range(0, len(ticklabs), 2):
        ticklabs[ii] = ''
    # ticklabs2 = ticklabs[::2]
    ax.set_xticklabels(ticklabs)
    ax.minorticks_off()
    # ax.legend()
    return ax


def plot_td_corr_data(time_intervals, data, ylabel="Error [dB]", ax=None, switchcolors=False):
    # Show octave spectrum
    if ax is None:
        ax = plt.gca()
    # if not switchcolors:
    #     colors = ['orangered', 'mediumorchid', 'violet', 'deeppink']
    # else:
    #     # colors = ['coral', 'royalblue', 'firebrick', 'seagreen']
    #     colors = ['k', 'royalblue', 'firebrick', 'coral']
    # linestyles = ['solid', 'dashed', (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    linestyles = ['solid', 'solid', 'solid', 'solid']
    for ii, key in enumerate(data.keys()):
        if ii == 0:
            lw = 2.5
        else:
            lw = 1.5
        x = np.nan_to_num(data[key])

        if data[key].ndim == 1:
            ax.plot(time_intervals, x, color=colors[ii], linestyle=linestyles[ii],
                    linewidth=lw, label=key, alpha=0.9)
        else:
            mean, std_minus, std_plus = mean_confidence_interval(x)
            ax.plot(time_intervals, mean, color=colors[ii], linestyle=linestyles[ii],
                    linewidth=lw, label=key, alpha=0.9)
            ax.fill_between(time_intervals, std_minus, std_plus, color=colors[ii], linestyle=linestyles[ii],
                            linewidth=lw, alpha=0.6)

    ax.grid(which='major', linestyle=':')
    # ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(r'Time [s]')
    ax.set_ylabel(ylabel)

    # ax.set_xticks(np.arange(0, len(r)))
    # ticklabs = [str(round(rr,2)) for rr in r]
    # for ii in range(0,len(ticklabs),2):
    #     ticklabs[ii] = ''
    # # ticklabs2 = ticklabs[::2]
    # ax.set_xticklabels(ticklabs)
    ax.minorticks_off()
    legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=4, mode="expand", borderaxespad=0., handlelength=1)
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    return ax


def plot_3rdoctave_band_box_plot(fig, data):
    # Show octave spectrum
    mosaic = """
    AAA
    BBB
    CCC
    DDD
    """
    axd = fig.subplot_mosaic(mosaic, sharex = True)
    iterable = ['A', 'B', 'C', 'D']
    ticks = PyOctaveBand._thirdoctave()
    ticks, _, _ = PyOctaveBand.getansifrequencies(3, limits=[30, 7999])
    keys = list(data.keys())
    truncate = data[keys[0]].shape[-1]
    # ticks = ticks[:truncate]
    # octave_ticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    # octave_tick_labels = ['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k']
    tick_labels = ['31.5','','', '63', '','','125','','', '250','','', '500','','', '1k','','', '2k','','', '4k','','', '8k']
    # for ii, tick in enumerate(ticks):
    #     if tick in octave_ticks:
    #         ind = np.argwhere(np.array(octave_ticks) == tick)
    #         tick_labels.append(octave_tick_labels[np.squeeze(ind)])
    #     else:
    #         tick_labels.append('')

    positions = np.arange(0, len(tick_labels))
    widths = [0.7, 0.7, 0.5]
    for ii, key in enumerate(data.keys()):
        ax = axd[iterable[ii]]
        for j in range(4,len(positions)):
            ax.boxplot(
                data[key][:, j],
                positions=np.array(positions[j][None]),
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=colors[ii]),
                notch=True,
                widths=0.7,
                whis=0,
            )

        # ax.set_yticks([0., 0.5, 1])
        ax.set_ylim([0., 1])
        # ax.set_ylim([25., 78])
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(which='major', linestyle=':')
        # ax.grid(which='minor', linestyle=':')
        ax.set_xlim([3, len(tick_labels)])
        # if ii == 3:
        #     ax.set_xlabel(r'Frequency [Hz]')
        # if ii != 3:
        #     ax.get_xaxis().set_ticklabels([])
    return fig


def plot_subplots_box_plot(fig, data, xticks):
    # Show octave spectrum
    mosaic = """
    AAA
    BBB
    CCC
    DDD
    """
    axd = fig.subplot_mosaic(mosaic, sharex = True)
    iterable = ['A', 'B', 'C', 'D']
    keys = list(data.keys())

    # tick_labels = ['31.5','','', '63', '','','125','','', '250','','', '500','','', '1k','','', '2k','','', '4k','','', '8k']
    # for ii, tick in enumerate(ticks):
    #     if tick in octave_ticks:
    #         ind = np.argwhere(np.array(octave_ticks) == tick)
    #         tick_labels.append(octave_tick_labels[np.squeeze(ind)])
    #     else:
    #         tick_labels.append('')

    positions = np.arange(0, len(xticks) +2)
    ticklabels = np.arange(0.0, 0.75, 0.05).round(2)
    ticklabels = list(ticklabels)
    ticklabels[::2] = ['']*len(ticklabels[::2])
    # ticklabels = [''] + list(ticklabels) + ['']
    for ii, key in enumerate(data.keys()):
        ax = axd[iterable[ii]]
        for j in positions[1:-1]:
            ax.boxplot(
                data[key][j-1],
                positions=np.array(positions[j][None]),
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=colors[ii]),
                notch=True,
                widths=0.6,
                whis=0,
            )

        # ax.set_yticks([0., 0.5, 1])
        ax.set_ylim([-28., -9.])
        # ax.set_ylim([0., 1])
        # ax.set_ylim([12., 50.])
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels)
        ax.grid(which='major', linestyle=':')
        # ax.grid(which='minor', linestyle=':')
        # ax.set_xlim([3, len(tick_labels)])
        # if ii == 3:
        #     ax.set_xlabel(r'Frequency [Hz]')
        # if ii != 3:
        #     ax.get_xaxis().set_ticklabels([])

    return fig

def save_rirs():
    normalize = lambda x : x/np.max(abs(x))
    # normalize = lambda x : x/np.linalg.norm(x)
    fs = 16000
    def preprocess(rir, reference_rir = None):
        rir = normalize(rir)
        freq = np.fft.rfftfreq(len(rir), d=1 / fs)
        freq_ind = np.argmin(freq < 200)
        if reference_rir is not None:
            reference_rir = normalize(reference_rir)
            fr_in = np.fft.rfft(reference_rir)
            fr_post = np.fft.rfft(rir)
            fr_post = np.concatenate((fr_in[:freq_ind], fr_post[freq_ind:]))
            rir_post = np.fft.irfft(fr_post)
            return rir_post
        else:
            return rir

    csgm = np.load(npzfilecsgm, allow_pickle=True)
    hifi = np.load(npzfilehifi, allow_pickle=True)
    segan = np.load(npzfilesegan, allow_pickle=True)
    N = len(csgm['responses_true'])
    true_RIRs = []
    input_RIRs = []
    CSGM_RIRs = []
    SEGAN_RIRs = []
    HiFi_RIRs = []
    for i in range(N):
        # HiFi input and ground truth have are normalized differently than the other two (SEGAN and CSGM)
        truerir = csgm['responses_true'][i]
        csgminput = csgm['responses_aliased'][i]
        csgmrir = csgm['responses_adCSGM'][i]
        # seganinput = segan['input_rirs'][i]
        # hifiinput = hifi['input_rirs'][i]
        seganrir = segan['G_rirs'][i]
        hifirir = hifi['G_rirs'][i]

        truerir = preprocess(truerir)
        csgmrir = preprocess(csgmrir, reference_rir=csgminput)
        seganrir = preprocess(seganrir, reference_rir=csgminput)
        hifirir = preprocess(hifirir, reference_rir=csgminput)
        csgminput = preprocess(csgminput, reference_rir=csgminput)

        true_RIRs.append(truerir)
        input_RIRs.append(csgminput)
        CSGM_RIRs.append(csgmrir)
        SEGAN_RIRs.append(seganrir)
        HiFi_RIRs.append(hifirir)

    newdict = dict(csgm)
    newdict['responses_true'] = np.array(true_RIRs)
    newdict['responses_aliased'] = np.array(input_RIRs)
    newdict['responses_adCSGM'] = np.array(CSGM_RIRs)
    np.savez(os.path.join(str(csgmfilepath.absolute()), 'generator_inference_processed'), **newdict)

    newdict = dict(segan)
    newdict['true_rirs'] = np.array(true_RIRs)
    newdict['input_rirs'] = np.array(input_RIRs)
    newdict['G_rirs'] = np.array(SEGAN_RIRs)
    # np.savez(seganfilepath + '/generator_inference_processed', **newdict)
    np.savez(os.path.join(str(seganfilepath.absolute()), 'generator_inference_processed'), **newdict)

    newdict = dict(csgm)
    newdict['true_rirs'] = np.array(true_RIRs)
    newdict['input_rirs'] = np.array(input_RIRs)
    newdict['G_rirs'] = np.array(HiFi_RIRs)
    # np.savez(hififilepath + '/generator_inference_processed', **newdict)
    np.savez(os.path.join(str(hififilepath.absolute()), 'generator_inference_processed'), **newdict)



def run_metrics():

    csgm = np.load(npzfilecsgmproc, allow_pickle=True)
    hifi = np.load(npzfilehifiproc, allow_pickle=True)
    segan = np.load(npzfileseganproc, allow_pickle=True)

    get_eval_metrics(csgm['responses_adCSGM'] + 1e-8 * np.random.randn(*csgm['responses_adCSGM'].shape),
                     csgm['responses_true'],
                     csgm['responses_aliased'], str(csgmfilepath))
    get_eval_metrics(hifi['G_rirs'], hifi['true_rirs'],
                     hifi['input_rirs'], str(hififilepath))
    get_eval_metrics(segan['G_rirs'], segan['true_rirs'],
                     segan['input_rirs'], str(seganfilepath))

def get_data():
    # npzfilecsgm = './CSGM/inference_data/generator_inference_processed.npz'
    # npzfilesegan = './SEGAN/generated_files/generator_inference_processed.npz'
    # npzfilehifi = './hifi-extension/generated_files/generator_inference_processed.npz'
    # npzfilecsgm = './CSGM/inference_data/inference_data.npz'
    # npzfilesegan = './SEGAN/generated_files/generator_inference_file.npz'
    # npzfilehifi = './hifi-extension/generated_files/generator_inference_file.npz'

    csgm = np.load(npzfilecsgm, allow_pickle=True)
    hifi = np.load(npzfilehifi, allow_pickle=True)
    segan = np.load(npzfilesegan, allow_pickle=True)
    return csgm, hifi, segan

def get_rirs(index=0):

    csgm = np.load(npzfilecsgmproc, allow_pickle=True)
    hifi = np.load(npzfilehifiproc, allow_pickle=True)
    segan = np.load(npzfileseganproc, allow_pickle=True)

    return csgm['responses_true'][index], csgm['responses_aliased'][index], csgm['responses_adCSGM'][index], \
           hifi['G_rirs'][index], segan['G_rirs'][index]


def plot_all_rirs(index=0):
    true_rirs, planewave, csgm, hifi, segan = get_rirs(index = index)
    mosaic = [['pw', 'pw',          'pw'],
              ['csgm', 'csgm',    'csgm'],
              ['hifi', 'hifi',    'hifi'],
              ['segan', 'segan', 'segan']]

    plot_settings(double_width=False)
    figsize = get_figsize(wf=2, hf=.6)
    # allrirs = [planewave, csgm, hifi, segan]
    # fig, ax = plt.subplots(figsize=(width, 3*width / 7))
    fig, ax_dict = plt.subplot_mosaic(mosaic, sharex=True, figsize=figsize)
    ax_dict['pw'] = plot_rir(true_rirs, planewave, label = 'PW', color = colors[0],
                             ax = ax_dict['pw'], t_intervals= [0.01, 0.15])
    ax_dict['csgm'] = plot_rir(true_rirs, csgm, label = 'CSGM', color = colors[1],
                               ax = ax_dict['csgm'], t_intervals= [0.01, 0.15])
    ax_dict['hifi'] = plot_rir(true_rirs, hifi, label = 'HiFiGAN', color = colors[2],
                               ax = ax_dict['hifi'], t_intervals= [0.01, 0.15])
    ax_dict['segan'] = plot_rir(true_rirs, segan, label = 'CGAN', color = colors[3],
                                ax = ax_dict['segan'], t_intervals= [0.01, 0.15])
    handles_pw, labels_pw = ax_dict['pw'].get_legend_handles_labels()
    handles_csgm, labels_csgm = ax_dict['csgm'].get_legend_handles_labels()
    handles_hifi, labels_hifi = ax_dict['hifi'].get_legend_handles_labels()
    handles_segan, labels_segan = ax_dict['segan'].get_legend_handles_labels()
    handles = handles_pw + [handles_csgm[1]] + [handles_hifi[1]] +  [handles_segan[1]]
    labels = labels_pw + [labels_csgm[1]] + [labels_hifi[1]] + [labels_segan[1]]
    # fig.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=5, mode="expand", borderaxespad=0., handlelength=1)

    legend = fig.legend(handles, labels,  bbox_to_anchor=(0.035, 0.9, 1., .102), loc='lower center',
               ncol=5, handlelength=0.8, handletextpad = 0.3,handleheight = 0.6)
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    fig.text(0.5, 0.015, 'Time [s]', fontsize=11, ha='center')
    # fig.text(0.0001, 0.35, 'Normalized Pressure [Pa]', fontsize=11, rotation='vertical')

    # fig.supxlabel('Time [s]', fontsize = 11)
    fig.supylabel('Normalized Pressure [Pa]', fontsize = 11)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    fig.show()
    # fig.savefig(figdir + f'/rirs_pos_{index}_thinLine.pdf', dpi = 300, bbox_inches='tight',pad_inches = 0)
    # fig.savefig(figdir + f'/rirs_pos_{index}.pdf', dpi = 300, bbox_inches='tight',pad_inches = 0)
def plot_all_frfs(index=0, normalize = True):
    true_rirs, planewave, csgm, hifi, segan = get_rirs(index = index)
    mosaic = [['pw', 'pw',          'pw'],
              ['csgm', 'csgm',    'csgm'],
              ['hifi', 'hifi',    'hifi'],
              ['segan', 'segan', 'segan']]

    figsize = plot_settings(double_width=True)
    # allrirs = [planewave, csgm, hifi, segan]
    # fig, ax = plt.subplots(figsize=(width, 3*width / 7))
    fig, ax_dict = plt.subplot_mosaic(mosaic, sharex=True, figsize=(figsize[0], .45*figsize[0]))
    ax_dict['pw'] = plot_frf(true_rirs, planewave, label = 'PW', color = colors[0],
                            normalize= normalize, ax = ax_dict['pw'])
    ax_dict['csgm'] = plot_frf(true_rirs, csgm, label = 'CSGM', color = colors[1],
                            normalize= normalize, ax = ax_dict['csgm'])
    ax_dict['hifi'] = plot_frf(true_rirs, hifi, label = 'HiFiGAN', color = colors[2],
                            normalize= normalize, ax = ax_dict['hifi'])
    ax_dict['segan'] = plot_frf(true_rirs, segan, label = 'CGAN', color = colors[3],
                                normalize= normalize, ax = ax_dict['segan'])
    handles_pw, labels_pw = ax_dict['pw'].get_legend_handles_labels()
    handles_csgm, labels_csgm = ax_dict['csgm'].get_legend_handles_labels()
    handles_hifi, labels_hifi = ax_dict['hifi'].get_legend_handles_labels()
    handles_segan, labels_segan = ax_dict['segan'].get_legend_handles_labels()
    handles = handles_pw + [handles_csgm[1]] + [handles_hifi[1]] +  [handles_segan[1]]
    labels = labels_pw + [labels_csgm[1]] + [labels_hifi[1]] + [labels_segan[1]]
    # fig.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=5, mode="expand", borderaxespad=0., handlelength=1)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    legend = fig.legend(handles, labels, bbox_to_anchor=(0., 0.99, 1.1, .102), loc='lower center',
                        ncol=5,  borderaxespad=0., handlelength=1)
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    fig.supxlabel('Frequency [Hz]', fontsize = 11)
    fig.supylabel('Normalised SPL [dB]', fontsize = 11)
    # fig.tight_layout()
    fig.show()
    # fig.savefig(figdir + f'/frfs_pos_{index}_thinLine.pdf', dpi = 300, bbox_inches='tight',pad_inches = 0)
    # fig.savefig(figdir + f'/frfs_pos_{index}.pdf', dpi = 300, bbox_inches='tight',pad_inches = 0)

def plot_broadband_error_dist(magnitude = True):
    indices = np.arange(0, 15)
    fs = 16000
    freq = np.fft.rfftfreq(16000, d=1 / fs)
    if magnitude:
        func = lambda x:  np.abs(x)
        lims = [-10, 10]
        ticks = [-5, 0, 5]

    else:
        func = lambda x: np.angle(x)
        lims = [-30, 10]
        ticks = [-25, -10, 5]
    def nse_broadband(x, y):

        # Compute the median of the non-zero elements
        m = np.median(y[y > 0])
        # Assign the median to the zero elements
        y[y == 0] = m
        return (x - y) ** 2 / (y) ** 2
    ae_broadband = lambda  x, y : abs(x - y)
    plotindB = lambda x : 10*np.log10(x)
    true_TFs = []
    pw_TFs = []
    csgm_TFs = []
    hifi_TFs = []
    segan_TFs = []
    for ind in indices:
        true_rirs, planewave, csgm, hifi, segan = get_rirs(index = ind)
        freq = np.fft.rfftfreq(len(true_rirs), d=1 / fs)

        fr_truth = np.fft.rfft(true_rirs)
        fr_pw = np.fft.rfft(planewave)
        fr_csgm = np.fft.rfft(csgm)

        fr_hifi =np.fft.rfft(hifi)
        fr_segan = np.fft.rfft(segan)

        true_TFs.append(func(fr_truth))
        pw_TFs.append(func(fr_pw))
        csgm_TFs.append(func(fr_csgm))
        hifi_TFs.append(func(fr_hifi))
        segan_TFs.append(func(fr_segan))

    true_TFs = np.array(true_TFs)
    pw_TFs = np.array(pw_TFs)
    csgm_TFs = np.array(csgm_TFs)
    hifi_TFs = np.array(hifi_TFs)
    segan_TFs = np.array(segan_TFs)

    pw_TFs_mean, pw_TFs_lower, pw_TFs_upper = mean_confidence_interval(ae_broadband(pw_TFs, true_TFs),
                                                                       confidence=0.95)
    csgm_TFs_mean, csgm_TFs_lower, csgm_TFs_upper = mean_confidence_interval(ae_broadband(csgm_TFs, true_TFs),
                                                                             confidence=0.95)
    hifi_TFs_mean, hifi_TFs_lower, hifi_TFs_upper = mean_confidence_interval(ae_broadband(hifi_TFs, true_TFs),
                                                                             confidence=0.95)
    segan_TFs_mean, segan_TFs_lower, segan_TFs_upper = mean_confidence_interval(ae_broadband(segan_TFs, true_TFs),
                                                                                confidence=0.95)

    mus = [plotindB(pw_TFs_mean), plotindB(csgm_TFs_mean), plotindB(hifi_TFs_mean), plotindB(segan_TFs_mean)]
    ci_lower = [plotindB(pw_TFs_lower), plotindB(csgm_TFs_lower), plotindB(hifi_TFs_lower), plotindB(segan_TFs_lower)]
    ci_upper = [plotindB(pw_TFs_upper), plotindB(csgm_TFs_upper), plotindB(hifi_TFs_upper), plotindB(segan_TFs_upper)]

    mosaic = [['pw', 'pw',          'pw'],
              ['csgm', 'csgm',    'csgm'],
              ['hifi', 'hifi',    'hifi'],
              ['segan', 'segan', 'segan']]

    plot_settings(double_width=False)
    fsize = get_figsize(wf=2, hf=.55)

    from matplotlib.ticker import FuncFormatter
    def kilos(x, pos):
        'The two args are the value and tick position'
        if x < 10:
            return None
        elif x == 100:
            return '%1.1fk' % (x * 1e-3)
        elif x == 500:
            return '%1.1fk' % (x * 1e-3)
        elif x == 5000:
            return '%1dk' % (x * 1e-3)
        # elif x >= 1000:
        #     return '%1.1fk' % (x * 1e-3)


    iterable = ['pw', 'csgm', 'hifi', 'segan']
    labels = ['PW', 'CSGM', 'HiFiGAN', 'CGAN']

    fig, ax_dict = plt.subplot_mosaic(mosaic, sharex=True, figsize=fsize)
    formatter = FuncFormatter(kilos)
    freq_ind = np.argwhere(freq > 100)[:, 0]
    for i in range(4):
        ax_dict[iterable[i]].semilogx(freq[freq_ind], mus[i][freq_ind], linewidth=1, color=colors[i], label = labels[i])
        ax_dict[iterable[i]].fill_between(freq[freq_ind], ci_lower[i][freq_ind], ci_upper[i][freq_ind], color=colors[i],
                        linewidth=1, alpha=0.4)
        # ax_dict[iterable[i]].axhline(y= mus[i][freq_ind].mean(), color = 'k', ls=  '--')
        # ax_dict[iterable[i]].text(1000,  mus[i][freq_ind].mean(), f'{round(mus[i][freq_ind].mean(), 3)}', fontsize=9, va='center', ha='center',
        #                           backgroundcolor='w')

        ax_dict[iterable[i]].set_xlim(freq[freq_ind][0], fs/2 + 1)


        ax_dict[iterable[i]].xaxis.set_minor_formatter(formatter)
        ax_dict[iterable[i]].xaxis.set_major_formatter(formatter)
        ax_dict[iterable[i]].set_yticks(ticks)
        ax_dict[iterable[i]].set_ylim(lims)
        ax_dict[iterable[i]].grid(linestyle=':', which='both', color='k')

    handles_pw, labels_pw = ax_dict['pw'].get_legend_handles_labels()
    handles_csgm, labels_csgm = ax_dict['csgm'].get_legend_handles_labels()
    handles_hifi, labels_hifi = ax_dict['hifi'].get_legend_handles_labels()
    handles_segan, labels_segan = ax_dict['segan'].get_legend_handles_labels()
    handles = handles_pw + [handles_csgm[0]] + [handles_hifi[0]] +  [handles_segan[0]]
    labels = labels_pw + [labels_csgm[0]] + [labels_hifi[0]] + [labels_segan[0]]
    # fig.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=5, mode="expand", borderaxespad=0., handlelength=1)
    # fig.subplots_adjust(wspace=0.05, hspace=0.15)

    legend = fig.legend(handles, labels, bbox_to_anchor=(0.73, 1.001),
           borderaxespad=0.,
           handlelength=0.8, handletextpad = 0.3,
           handleheight = 0.6, columnspacing = 1.,
           bbox_transform = plt.gcf().transFigure,
           ncol=4, frameon=False,
           fontsize = 11)
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    fig.text(0.55, 0.05, 'Frequency [Hz]', fontsize=11, ha='center')
    fig.text(0.005, 0.3, 'Absolute Error [dB]', fontsize=11, rotation='vertical')

    fig.tight_layout()
    fig.subplots_adjust(wspace=None, hspace=0.1)

    if magnitude:
        figlabel = 'mag'
    else:
        figlabel = 'phase'
    fig.show()

    # fig.savefig(figdir + f'/error_broadband_{figlabel}.pdf' , dpi = 300, bbox_inches='tight',pad_inches = 0)

# """ Plot fd metrics"""
def fdmetric(metric, quantity):
    response_numbers = np.arange(0, 15)
    PWdata = []
    SEGANdata = []
    HiFidata = []
    CSGMdata = []
    # npzfilepath = './hifi-extension/generated_files/generator_inference_processed.npz'
    responseshifi = np.load(npzfilehifiproc, allow_pickle= True)
    grid_ref = responseshifi['grid_ref']
    r = np.linalg.norm(grid_ref[:2], axis = 0)

    for response_number in response_numbers:
        csgmfilepath
        h5path1 = os.path.join(str(seganfilepath),f'metrics_inference_{response_number}.h5')
        h5path2 = os.path.join(str(hififilepath),f'metrics_inference_{response_number}.h5')
        h5path3 = os.path.join(str(csgmfilepath),f'metrics_inference_{response_number}.h5')
        # h5path1 = f'./SEGAN/generated_files/metrics_inference_{response_number}.h5'
        # h5path2 = f'./hifi-extension/generated_files/metrics_inference_{response_number}.h5'
        # h5path3 = f'./CSGM/inference_data/metrics_inference_{response_number}.h5'
        metrics_SEGAN = config_metrics(h5path1)
        metrics_hifi = config_metrics(h5path2)
        metrics_csgm = config_metrics(h5path3)
        PWdata.append(metrics_SEGAN[f'fd_plwav{metric}_{quantity}_{response_number}'])
        SEGANdata.append(metrics_SEGAN[f'fd_gan{metric}_{quantity}'], )
        HiFidata.append(metrics_hifi[f'fd_gan{metric}_{quantity}'])
        CSGMdata.append(metrics_csgm[f'fd_gan{metric}_{quantity}'])
    plot_dict = {
        'PW': np.squeeze(PWdata),
        'HiFiGAN': np.squeeze(HiFidata),
        'CGAN': np.squeeze(SEGANdata),
        'CSGM': np.squeeze(CSGMdata)
    }

    return plot_dict, r
