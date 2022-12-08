import numpy as np
import matplotlib.pyplot as plt
import librosa
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
from scipy.io import loadmat

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# def load_mat_resample(path, fs_new):
#     from scipy.io import loadmat
#     mat_dict = loadmat(path)
#     T = mat_dict['T'].min()
#     fs_old = mat_dict['fs'].min()
#     rirs = mat_dict['h'].T
#     grid = mat_dict['r'].T
#     rir_set_out = []
#     for rir in rirs:
#         rir_set_out.append(librosa.resample(rir, fs_old, fs_new))
#
#     return np.asarray(rir_set_out), grid, T

def to_db(signal, normalize = True):
    if normalize:
        signal /= np.max(abs(signal), axis = -1)
    dBout = 20*np.log10(abs(signal))
    return dBout


def freq_resp(freq, amp, in_db=True, smoothing_n=None, title=None, ls = '-', clr = 'k',
              labels=None, xlim=(10, 10000), ylim=(-60, 0), ax = None, alpha = 1):
    """ Plot amplitude of frequency response over time frequency f.
    Parameters
    ----------
    f : frequency array
    amp : array_like, list of array_like
    """
    # amp = amp/max(abs(amp))
    if not isinstance(amp, (list, tuple)):
        amp = [amp]
    if labels is not None:
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

    assert(all(len(a) == len(freq) for a in amp))
    # normalise
    if in_db:
        # Avoid zeros in spec for dB
        amp = [to_db(a) for a in amp]

    if smoothing_n is not None:
        smoothed = []
        for a in amp:
            smooth = np.zeros_like(a)
            for idx in range(len(a)):
                k_lo = idx / (2**(1/(2*smoothing_n)))
                k_hi = idx * (2**(1/(2*smoothing_n)))
                smooth[idx] = np.mean(a[np.floor(k_lo).astype(int):
                                        np.ceil(k_hi).astype(int) + 1])
            smoothed.append(smooth)
        amp = smoothed

    # fig, _ = plt.subplots()
    if ax == None:
        ax = plt.gca()
    [ax.semilogx(freq, a.flat, alpha = alpha, ls = ls, color = clr, linewidth = 2) for a in amp]

    if title is not None:
        plt.title(title)
    if smoothing_n is not None:
        if labels is None:
            labels = [None] * len(amp)
        # fake line for extra legend entry
        ax.plot([], [], '*', color='black')
        labels.append(r"$\frac{%d}{8}$ octave smoothing" % smoothing_n)
    if labels is not None:
        ax.legend(labels)

    # ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel('Magnitude [dB]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(ls = ':', color = 'k', alpha = 0.6, which = 'both')
    return ax

def transfer_function(freq, H, title=None, xlim=(10, 25000), xlabel = '',
                      ax = None, alpha = 1., fontsize= 14, signal_titles = None):
    # import matplotlib
    # plt.rc('text', usetex=True)
    # matplotlib.rcParams['text.usetex']=True
    # matplotlib.rcParams['text.latex.unicode']=True
    """Plot transfer function H (magnitude and phase) over time frequency f."""
    # fig, ax1 = plt.subplots()
    ax1 = ax
    if H.ndim > 1:
        N = H.shape[0]
        if signal_titles is None:
            signal_titles = np.empty(shape = (N,), dtype = str)
            signal_titles[:N] = ''
    else: N = 1
    if ax1 is None:
        ax1 = plt.gca()

    colors_magn = cm.Blues(np.linspace(0.4,1, N))
    colors_phase = cm.Oranges(np.linspace(0.3,1, N))

    # H += 10e-15
    for ii in range(N):
        ax1.semilogx(freq, to_db(H[ii]),
                     color=colors_magn[ii],
                     label='Magnitude {}'.format(signal_titles[ii]))
        ax1.set_ylabel('Magnitude [dB]', fontsize = (fontsize - 2))
        ax1.set_xlabel(xlabel, fontsize = (fontsize - 2))
        ax1.set_xlim(xlim)
        if ii < 1:
            ax2 = ax1.twinx()
        ax2.semilogx(freq, np.unwrap(np.angle(H[ii])),
                     color=colors_phase[ii],
                     label='Phase {}'.format(signal_titles[ii]), zorder=0, alpha = alpha)
        ax2.set_ylabel('Phase [rad]', fontsize = (fontsize - 2))


    # ax1.grid(True)
    ax1.grid('.', which='major')
    ax1.grid('.', which='minor')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
    if title is not None:
        if not isinstance(title, (list, tuple)):
            plt.title(title, fontsize = fontsize)
        else:
            plt.title(title[0], fontsize = fontsize)
    if ax is not None:
        ax = ax2
        return ax


def load_mat_grid(path):
    mat_dict = loadmat(path)
    # T = mat_dict['T'].min()
    # fs_old = mat_dict['fs'].min()
    # rirs = mat_dict['h'].T
    grid = mat_dict['r']
    # rir_set_out = []
    # for rir in rirs:
    #     rir_set_out.append(librosa.resample(rir, fs_old, fs_new))

    return grid.T

def cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.

    .. math::

        \phi = \arctan \left( \frac{y}{x} \right) \\
        \theta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

    with :math:`\phi \in [-pi, pi], \theta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates

    Returns
    -------
    phi : float or `numpy.ndarray`
            Azimuth angle in radiants
    theta : float or `numpy.ndarray`
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius

    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return np.array([phi, theta, r])


def sph2cart(alpha, beta, r):
    r"""Spherical to cartesian coordinate transform.

    .. math::

        x = r \cos \alpha \sin \beta \\
        y = r \sin \alpha \sin \beta \\
        z = r \cos \beta

    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`

    Parameters
    ----------
    alpha : float or array_like
            Azimuth angle in radiants
    beta : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius

    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates

    """
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return np.array([x, y, z])



data_object = np.load('generator_inference.npz')

G_rirs = data_object['G_rirs']
true_rirs = data_object['true_rirs']
input_rirs = data_object['input_rirs']

# FFTs

G_TF = np.fft.rfft(G_rirs, axis = -1)
true_TF = np.fft.rfft(true_rirs, axis = -1)
input_TF = np.fft.rfft(input_rirs, axis = -1)
corrG = []
error_G = []
error_G_TF = []

# G with ground truth
for ii in range(len(G_rirs)):
    corrG.append(np.corrcoef(G_rirs[ii], true_rirs[ii])[0,1])
    sq_error = (true_rirs[ii] - G_rirs[ii])**2
    sq_error_TF = abs(true_TF[ii] - G_TF[ii])**2
    error_G.append(sq_error)
    error_G_TF.append(sq_error_TF)

# input (aliased) with ground truth
error_G = np.asarray(error_G)
error_G_TF = np.asarray(error_G_TF)
corr_in = []
error_in = []
error_in_TF = []

for ii in range(len(G_rirs)):
    corr_in.append(np.corrcoef(input_rirs[ii], true_rirs[ii])[0,1])
    sq_error = (true_rirs[ii] - input_rirs[ii])**2
    error_in.append(sq_error)
    sq_error_TF = abs(true_TF[ii] - input_TF[ii])**2
    error_in_TF.append(sq_error_TF)

error_in = np.asarray(error_in)
error_in_TF = np.asarray(error_in_TF)

fs = 16000
t = np.linspace(0, G_rirs.shape[-1]/fs, G_rirs.shape[-1])
freq = np.fft.rfftfreq(len(t), d = 1/fs)




fig = plt.figure(figsize=(9,3.5))
axd = fig.subplot_mosaic(
    """
    AC
    .D
    BE
    """
)


# fig, (ax1, ax2) = plt.subplots(2,1, figsize = (7,3.5), sharex = True)
for jj in range(len(error_G)):
    axd['B'].plot(t[:int(0.4*fs)], np.sqrt(error_G[jj, :int(0.4*fs)]), color = 'lightsalmon', alpha = 0.4, linewidth = 0.5)
meanplot = axd['B'].plot(t[:int(0.4*fs)], np.sqrt(error_G).mean(axis = 0)[:int(0.4*fs)], color = 'r', linewidth = 2, label = 'Mean input')
salmon_patch = mpatches.Patch(color='lightsalmon', label='Error realizations input')

axd['B'].set_ylabel(r'$\sqrt{(p_{true_{i}}(t) - p_{input_{i}}(t))^2}$')
# ax1.set_xlabel('time [s]')
axd['B'].legend(handles = [salmon_patch, meanplot[0]])
axd['B'].grid(ls = ':', color = 'k', alpha = 0.7)

for jj in range(len(error_in)):
    axd['A'].plot(t[:int(0.4*fs)], np.sqrt(error_in[jj, :int(0.4*fs)]), color = 'cornflowerblue', alpha = 0.4, linewidth = 0.5)
meanplot = axd['A'].plot(t[:int(0.4*fs)], np.sqrt(error_in).mean(axis = 0)[:int(0.4*fs)], color = 'darkslateblue', linewidth = 2, label = 'Mean G')
cornflowerblue_patch = mpatches.Patch(color='cornflowerblue', label='Error realizations G')

axd['A'].set_ylabel(r'$\sqrt{(p_{true_{i}}(t) - p_{G_{i}}(t))^2}$')
# ax1.set_xlabel('time [s]')
axd['A'].legend(handles = [cornflowerblue_patch, meanplot[0]])
axd['A'].grid(ls = ':', color = 'k', alpha = 0.7)
axd['A'].xaxis.set_ticklabels([])
axd['B'].set_xlabel('Time [s]')
# fig.show()

# fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (7,3.5), sharex = True)


for jj in range(len(input_rirs)):
    axd['C'].plot(t[:int(0.4*fs)], G_rirs[jj, :int(0.4*fs)], color = 'steelblue', alpha = 0.5, linewidth = 0.5)
    axd['D'].plot(t[:int(0.4*fs)], input_rirs[jj, :int(0.4*fs)], color = 'tomato', alpha = 0.5, linewidth = 0.5)
    axd['E'].plot(t[:int(0.4*fs)], true_rirs[jj, :int(0.4*fs)], color = 'olive', alpha = 0.5, linewidth = 0.5)

meanplot = axd['C'].plot(t[:int(0.4*fs)], G_rirs.mean(axis = 0)[:int(0.4*fs)],
                         color = 'darkslategrey', linewidth = 2, label = 'Mean G')
meanplot = axd['D'].plot(t[:int(0.4*fs)], input_rirs.mean(axis = 0)[:int(0.4*fs)],
                         color = 'firebrick', linewidth = 2, label = 'Mean G')
meanplot = axd['E'].plot(t[:int(0.4*fs)], true_rirs.mean(axis = 0)[:int(0.4*fs)],
                         color = 'darkolivegreen', linewidth = 2, label = 'Mean G')

steelblue_patch = mpatches.Patch(color='steelblue', label='Generator Output')
tomato_patch = mpatches.Patch(color='tomato', label='Generator input')
olive_patch = mpatches.Patch(color='olive', label='True RIRs')

axd['C'].xaxis.set_ticklabels([])
axd['D'].xaxis.set_ticklabels([])

axd['C'].grid(which = 'both', ls = ':', color = 'k', alpha = 0.7)
axd['D'].grid(which = 'both', ls = ':', color = 'k', alpha = 0.7)
axd['E'].grid(which = 'both', ls = ':', color = 'k', alpha = 0.7)

axd['D'].set_ylabel("Normalized Pressure")
axd['E'].set_xlabel("Time [s]")

axd['A'].get_shared_x_axes().join(axd['A'], axd['B'])
axd['C'].get_shared_x_axes().join(axd['C'], axd['D'])
axd['C'].get_shared_x_axes().join(axd['C'], axd['E'])

fig.legend(handles = [steelblue_patch, tomato_patch, olive_patch], loc = 'upper right')

fig.savefig('RIR_results.png', dpi = 300)

# plot with grid
grid = load_mat_grid('/Users/xen/PhD Acoustics/Repositories/hifi-extension/plane wave responses/PWdata/rir_iec_plane.mat')
grid[-1] -= 0.628
grid_sphere = cart2sph(grid[0],grid[1],grid[2] )
indx_sorted = np.argsort(grid_sphere[-1])
grid_new = grid[0:2, indx_sorted]
corrG_sorted = np.asarray(corrG)[indx_sorted]
corr_in_sorted = np.asarray(corr_in)[indx_sorted]
fig, ax = plt.subplots(1,2,  sharey= True, figsize=(8.4,3.8))
sc1 = ax[0].scatter(grid_new[0], grid_new[1], c = corrG_sorted, s = 100, cmap='viridis', vmin = 0.2, vmax = 1)
ax[0].set_title('Generator Output')
ax[0].set_xlabel('x [m]')
ax[1].set_xlabel('x [m]')
ax[0].set_ylabel('y [m]')
sc2 = ax[1].scatter(grid_new[0], grid_new[1], c = corr_in_sorted, s = 100, cmap='viridis',  vmin = 0.2, vmax = 1)
ax[1].set_title('Generator Input')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(sc1, cax=cbar_ax)
fig.savefig('corr_results.png', dpi = 300)

# fig, ax = plt.subplots(1,1, figsize = (7,2.5))
# transfer_function(freq,  np.c_[G_TF[300], true_TF[300]].T, ax = ax, xlim = (10, 8000),
#                   signal_titles= ['G', 'True'], xlabel = 'Frequency [Hz]' )
#
# fig.show()

# # G realizations
# fig, ax = plt.subplots(1,1, figsize = (7,2.5))
# freq_mean = np.mean(G_TF, axis = 0)
# for ii in range(len(G_TF)):
#     freq_resp(freq, G_TF[ii], ax = ax, xlim = (10, 8000),  ylim = (-80, 5), alpha = 0.5, clr = 'steelblue')
#
# freq_resp(freq, freq_mean, ax = ax, xlim = (10, 8000), ylim = (-80, 5),  clr = 'darkslategrey')
# fig.show()
#
# # input realizations
# fig, ax = plt.subplots(1,1, figsize = (7,2.5))
# freq_mean = np.mean(input_TF, axis = 0)
# for ii in range(len(input_TF)):
#     freq_resp(freq, input_TF[ii], ax = ax, xlim = (10, 8000),  ylim = (-80, 5), alpha = 0.5, clr = 'tomato')
#
# freq_resp(freq, freq_mean, ax = ax, xlim = (10, 8000), ylim = (-80, 5), clr = 'firebrick')
# fig.show()



# fig, ax = plt.subplots(3,1, sharex = True, figsize = (7,5))

fig = plt.figure(figsize=(9,4))
axd = fig.subplot_mosaic(
    """
    AC
    .D
    BE
    """
)
# G error

freq_mean = np.mean(error_G_TF, axis = 0)
for ii in range(len(G_TF)):
    freq_resp(freq, error_G_TF[ii], ax = axd['A'], xlim = (10, 8000),  ylim = (-130, 10),
              alpha = 0.5, clr = 'cornflowerblue')

freq_resp(freq, freq_mean, ax = axd['A'], xlim = (10, 8000), ylim = (-130, 10),  clr = 'darkslateblue')

freq_mean = np.mean(error_in_TF, axis = 0)
for ii in range(len(G_TF)):
    freq_resp(freq, error_in_TF[ii], ax = axd['B'], xlim = (10, 8000),  ylim = (-130, 10), alpha = 0.5, clr = 'lightsalmon')

freq_resp(freq, freq_mean, ax = axd['B'], xlim = (10, 8000), ylim = (-130, 10),  clr = 'r')

cornflower_patch = mpatches.Patch(color='cornflowerblue', label='Generator Error')
salmon_patch = mpatches.Patch(color='lightsalmon', label='Input Error')
# green_patch = mpatches.Patch(color='olive', label='True Responses')

axd['A'].set_ylabel(r'$\sqrt{(p_{true_{i}}(\omega) - p_{input_{i}}(\omega))^2}$' + '\n[SPL - dB re 1 Pa]')
axd['B'].set_ylabel(r'$\sqrt{(p_{true_{i}}(\omega)  - p_{G_{i}}(\omega))^2}$' + '\n[SPL - dB re 1 Pa]')
# ax1.set_xlabel('time [s]')
axd['A'].legend(handles = [cornflower_patch, salmon_patch], loc = 'lower right')
axd['A'].xaxis.set_ticklabels([])
axd['B'].set_xlabel('Frequency [Hz]')



# Grealizations
freq_mean = np.mean(G_TF, axis = 0)
for ii in range(len(G_TF)):
    freq_resp(freq, G_TF[ii], ax = axd['C'], xlim = (10, 8000),  ylim = (-80, 5), alpha = 0.5, clr = 'steelblue')

freq_resp(freq, freq_mean, ax = axd['C'], xlim = (10, 8000), ylim = (-80, 5),  clr = 'darkslategrey')

# input realizations
freq_mean = np.mean(input_TF, axis = 0)
for ii in range(len(input_TF)):
    freq_resp(freq, input_TF[ii], ax = axd['D'], xlim = (10, 8000),  ylim = (-80, 5), alpha = 0.5, clr = 'tomato')

freq_resp(freq, freq_mean, ax = axd['D'], xlim = (10, 8000), ylim = (-80, 5), clr = 'firebrick')

# true realizations
freq_mean = np.mean(true_TF, axis = 0)
for ii in range(len(true_TF)):
    freq_resp(freq, true_TF[ii], ax = axd['E'], xlim = (10, 8000),  ylim = (-80, 5), alpha = 0.5, clr = 'olive')

freq_resp(freq, freq_mean, ax = axd['E'], xlim = (10, 8000), ylim = (-80, 5), clr = 'darkolivegreen')

blue_patch = mpatches.Patch(color='steelblue', label='Generator Outputs')
red_patch = mpatches.Patch(color='tomato', label='Generator Inputs')
green_patch = mpatches.Patch(color='olive', label='True Responses')

axd['C'].xaxis.set_ticklabels([])
axd['D'].xaxis.set_ticklabels([])
axd['D'].set_ylabel("SPL [dB ref 1 Pa] ")
axd['E'].set_xlabel("Frequency [Hz]")

axd['A'].get_shared_x_axes().join(axd['A'], axd['B'])
axd['C'].get_shared_x_axes().join(axd['C'], axd['D'])
axd['C'].get_shared_x_axes().join(axd['C'], axd['E'])

fig.legend(handles = [blue_patch, red_patch, green_patch], loc = 'lower right')
# fig.show()


# ax[1].set_ylabel("Magnitude [dB]")
# ax[2].set_xlabel("Frequency [Hz]")


fig.savefig('TF_results.png', dpi = 300)

