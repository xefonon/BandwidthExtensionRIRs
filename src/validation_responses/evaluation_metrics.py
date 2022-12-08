import numpy as np
from scipy.spatial import distance
from src import PyOctaveBand
from tqdm import tqdm
import h5py
import os

def get_eval_metrics(G_rirs, true_rirs, input_rirs, output_dir):
    N = len(G_rirs)
    pbar2 = tqdm(range(N))

    for i in pbar2:
        output_path = os.path.join(output_dir, f'metrics_inference_{i}.h5')
        hf = h5py.File(output_path, 'w')

        td_metrics_gen, time_intervals = time_metrics(G_rirs[i],
                                                      true_rirs[i],
                                                      fs=16000,resolution = 1e-2, t_0=0, t_end=250e-3)
        for k in td_metrics_gen.keys():
            hf['td_gan'+k]=td_metrics_gen[k]

        td_metrics_plwav, _ = time_metrics(input_rirs[i],
                                           true_rirs[i],
                                           fs=16000,resolution = 1e-2, t_0=0, t_end=250e-3)
        for k in td_metrics_plwav.keys():
            hf['td_plwav'+k]=td_metrics_plwav[k]

        fd_metrics_gen, freq = freq_metrics(G_rirs[i],
                                            true_rirs[i],
                                            fs=16000)
        for k in fd_metrics_gen.keys():
            hf['fd_gan'+k]=fd_metrics_gen[k]

        fd_metrics_plwav, _ = freq_metrics(input_rirs[i],
                                           true_rirs[i],
                                           fs=16000, index=i)
        for k in fd_metrics_plwav.keys():
            hf['fd_plwav'+k]=fd_metrics_plwav[k]

        # octave_band_metrics_gen, bands = octave_band_metrics(G_rirs[i],
        #                                                      true_rirs[i],
        #                                                      fs=16000)
        # for k in octave_band_metrics_gen.keys():
        #     hf['octave_gan'+k]=octave_band_metrics_gen[k]

        # octave_band_metrics_plwav, _ = octave_band_metrics(input_rirs[i],
        #                                                    true_rirs[i],
        #                                                    fs=16000)

        # for k in octave_band_metrics_plwav.keys():
        #     hf['octave_plwav'+k]=octave_band_metrics_plwav[k]

        pbar2.set_description(f"Getting metrics for response : {i + 1}/{N}")

        hf['time_intervals'] = time_intervals
        hf['freq'] = freq
        # hf['bands'] = bands
        hf.close()


def time_metrics(h, h_ref, fs, resolution=2e-3, t_0=5e-3, t_end=200e-3, index = None):
    performance = dict()
    intervals = np.arange(0, h.shape[0])[int(fs * t_0) : int(fs * t_end)][:: int(fs * resolution)]
    window = int(fs * resolution)
    nmse = []
    corr = []
    coh = []
    nmse_full = []
    corr_full = []
    coh_full = []
    mae_full = []
    mae = []
    for i in intervals:
        nmse.append(10 * np.log10(np.mean((h[i:i + window] - h_ref[i]) ** 2 / (h_ref[i:i + window]) ** 2)))
        mae.append(10 * np.log10(np.mean(np.abs(h[i:i + window] - h_ref[i]))))
        corr.append(np.corrcoef(h[i:i + window], h_ref[i:i + window])[0, 1])
        coh.append(1 - distance.cosine(h[i:i + window], h_ref[i:i + window]))
        # Full metrics until interval
    nmse_full.append(
        10 * np.log10(np.mean((h - h_ref) ** 2 / (h_ref) ** 2))
    )
    corr_full.append(np.corrcoef(h.flatten(), h_ref.flatten())[0, 1])
    coh_full.append(1 - distance.cosine(h.flatten(), h_ref.flatten()))
    mae_full.append(10 * np.log10(np.mean(np.abs(h - h_ref))))

    if index is None:
        performance["nmse"] = np.array(nmse)
        performance["mae"] = np.array(mae)
        performance["mae_full"] = np.array(mae_full)
        performance["corr"] = np.array(corr)
        performance["cos"] = np.array(coh)
        performance["nmse_full"] = np.array(nmse_full)
        performance["corr_full"] = np.array(corr_full)
        performance["cos_full"] = np.array(coh_full)
    else:
        performance[f"nmse_{index}"] = np.array(nmse)
        performance[f"mae_full_{index}"] = np.array(mae_full)
        performance[f"corr_{index}"] = np.array(corr)
        performance[f"cos_{index}"] = np.array(coh)
        performance[f"nmse_full_{index}"] = np.array(nmse_full)
        performance[f"corr_full_{index}"] = np.array(corr_full)
        performance[f"cos_full_{index}"] = np.array(coh_full)
    return (performance, intervals / fs)

def freq_metrics(h, h_ref, fs, nfft = 8193, index = None):
    freq = np.fft.rfftfreq(n = 2*(nfft-1), d = 1/fs)
    freq_ind = np.argmin(freq < 7e3)
    frstart = 500
    performance = dict()
    H = np.fft.rfft(h, n = 2*(nfft-1))
    H_ref = np.fft.rfft(h_ref, n = 2*(nfft-1))
    Hmag = abs(H)
    Hphase = np.angle(H)
    H_ref_mag = abs(H_ref)
    H_ref_phase = np.angle(H_ref)

    nmse_mag = 10 * np.log10(np.mean((Hmag[frstart:freq_ind] - H_ref_mag[frstart:freq_ind]) ** 2 / (H_ref_mag[frstart:freq_ind]) ** 2))
    corr_mag = np.corrcoef(Hmag[frstart:freq_ind], H_ref_mag[frstart:freq_ind])[0, 1]
    coh_mag = 1 - distance.cosine(Hmag[frstart:freq_ind], H_ref_mag[frstart:freq_ind])

    mae_mag = 10 * np.log10(np.mean(np.abs(Hmag[frstart:freq_ind] - H_ref_mag[frstart:freq_ind])))

    nmse_phase = 10 * np.log10(np.mean((Hphase[frstart:freq_ind] - H_ref_phase[frstart:freq_ind]) ** 2 / (H_ref_phase[frstart:freq_ind]) ** 2))
    corr_phase = np.corrcoef(Hphase[frstart:freq_ind], H_ref_phase[frstart:freq_ind])[0, 1]
    coh_phase = 1 - distance.cosine(Hphase[frstart:freq_ind], H_ref_phase[frstart:freq_ind])
    mae_phase = 10 * np.log10(np.mean(np.abs(Hphase[frstart:freq_ind] - H_ref_phase[frstart:freq_ind])))
    if index is None:
        performance["nmse_mag"] = np.array(nmse_mag)
        performance["mae_mag"] = np.array(mae_mag)
        performance["corr_mag"] = np.array(corr_mag)
        performance["cos_mag"] = np.array(coh_mag)
        performance["nmse_phase"] = np.array(nmse_phase)
        performance["mae_phase"] = np.array(mae_phase)
        performance["corr_phase"] = np.array(corr_phase)
        performance["cos_phase"] = np.array(coh_phase)
    else:
        performance[f"nmse_mag_{index}"] = np.array(nmse_mag)
        performance[f"mae_mag_{index}"] = np.array(mae_mag)
        performance[f"corr_mag_{index}"] = np.array(corr_mag)
        performance[f"cos_mag_{index}"] = np.array(coh_mag)
        performance[f"nmse_phase_{index}"] = np.array(nmse_phase)
        performance[f"mae_phase_{index}"] = np.array(mae_phase)
        performance[f"corr_phase_{index}"] = np.array(corr_phase)
        performance[f"cos_phase_{index}"] = np.array(coh_phase)

    return (performance, freq)

def octave_band_metrics(h, h_ref, fs, fmin=12, fmax =7999, index = None):
    performance = dict()
    _, bands, h_1_3rd = PyOctaveBand.octavefilter(h, fs=fs, fraction=3, order=6, limits=[fmin, fmax], show=0, sigbands=1)
    _, bands, href_1_3rd = PyOctaveBand.octavefilter(h_ref, fs=fs, fraction=3, order=6, limits=[fmin, fmax], show=0, sigbands=1)
    nmse = []
    corr = []
    coh = []
    mae = []
    for ind, band in enumerate(bands):
        nmse.append(10 * np.log10(np.mean((h_1_3rd[ind] - href_1_3rd[ind]) ** 2 / (href_1_3rd[ind]) ** 2)))
        corr.append(np.corrcoef(h_1_3rd[ind], href_1_3rd[ind])[0, 1])
        coh.append(1 - distance.cosine(h_1_3rd[ind], href_1_3rd[ind]))
        mae.append(10 * np.log10(np.mean(np.abs(h_1_3rd[ind] - href_1_3rd[ind]))))
    if index is None:
        performance["nmse"] = np.array(nmse)
        performance["mae"] = np.array(mae)
        performance["corr"] = np.array(corr)
        performance["cos"] = np.array(coh)
    else:
        performance[f"nmse_{index}"] = np.array(nmse)
        performance[f"mae_{index}"] = np.array(mae)
        performance[f"corr_{index}"] = np.array(corr)
        performance[f"cos_{index}"] = np.array(coh)

    return (performance, bands)

