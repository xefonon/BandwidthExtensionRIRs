from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import torch
from librosa.util import normalize
import numpy as np
from generator import Generator
from utils import config_from_yaml
import click
from utils import find_files
from pathlib import Path
from tqdm.contrib import tzip
import sys
sys.path.insert(0, "../../")
from validation_responses.evaluation_metrics import time_metrics, freq_metrics, octave_band_metrics
import os
from tqdm import tqdm
import h5py
device = 'gpu'

'./checkpoints_generator/G_Final_1.0'

def select_responses(ref_responses, rec_responses, grid_ref, return_indices = True):
    hsph = 0.628
    r = np.linalg.norm(grid_ref - np.array([[0., 0., hsph]]).T, axis=0)
    # radii = np.linspace(0.05, 0.7, n_selected)
    radii = np.unique(np.round(r, 3))
    radii[0] = 1e-8
    radii.sort()
    selected_grid = []
    selected_ref_responses = []
    selected_rec_responses = []
    indices = []
    np.random.seed(1234)
    for i, radius in enumerate(radii):
        prev_radius = radii[i-1] if i > 0 else 1e-18
        indx = np.squeeze(
            np.argwhere((r>prev_radius) & (r<=radius)))
        selected_indx = np.random.choice(indx, 1, replace=False) if len(indx) > 1 else indx
        selected_ref_responses.append(ref_responses[selected_indx])
        selected_rec_responses.append(rec_responses[selected_indx])
        selected_grid.append(grid_ref[:, selected_indx])
        indices.append(selected_indx)
    if return_indices:
        return np.squeeze(indices)
    else:
        return np.array(selected_ref_responses), np.array(selected_rec_responses), np.array(selected_grid)


def find_validation_data(root_dir, query="*.hdf5", return_grid=False):
    file_name = find_files(root_dir, query=query)
    if return_grid:
        with h5py.File(file_name[0], "r") as f:
            valid_true = f['rirs_ref'][:]
            valid_recon = f['rir_ridge'][:]
            grid_ref = f['grid_ref'][:]
        f.close()
        return valid_true, valid_recon, grid_ref
    else:
        with h5py.File(file_name[0], "r") as f:
            valid_true = f['rirs_ref'][:]
            valid_recon = f['rir_ridge'][:]
        f.close()

        return valid_true, valid_recon


def load_checkpoint(filepath, device):
    # assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(validation_dir, output_dir, checkpoint_dir, hp, with_postnet=False):
    generator = Generator(hp.in_channels, hp.out_channels, num_layers= hp.G_layers,
                                          num_stacks= hp.num_stacks, residual_channels= hp.residual_channels,
                                          gate_channels= hp.gate_channels,
                                          use_spectral_norm=hp.G_use_spectral_norm).to(device).to(device)


    checkpoint_path = checkpoint_dir + '/G_no_adversarial_200000iters'
    state_dict_g = load_checkpoint(checkpoint_path, device)
    generator.load_state_dict(state_dict_g['generator'])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.eval()

    valid_true_all, valid_recon_all, grid_ref = find_validation_data(validation_dir, return_grid= True) #shape: [709, 16384]

    indices = select_responses(valid_true_all, valid_recon_all, grid_ref,return_indices= True)
    valid_true_all = valid_true_all[indices]
    valid_recon_all = valid_recon_all[indices]
    grid_ref = grid_ref[:, indices]
    true_rirs = []
    input_rirs = []
    generator_rirs = []
    with torch.no_grad():
        for valid_true, valid_recon in tzip(valid_true_all, valid_recon_all):
            # true_rir = normalize(valid_true) * 0.95
            # recon_rir = normalize(valid_recon) * 0.95
            true_rir = valid_true
            recon_rir = valid_recon

            true_rirs.append(true_rir)
            input_rirs.append(recon_rir)

            recon_rir = torch.FloatTensor(recon_rir)
            recon_rir = recon_rir.reshape((1, 1, recon_rir.shape[-1])).to(device)

            y_generator, y_g_postnet = generator(recon_rir, with_postnet)
            G_rir = y_generator.reshape((y_generator.shape[2],))
            G_rir = G_rir.cpu().numpy()
            generator_rirs.append(G_rir)

    output_file = os.path.join(
        output_dir, 'generator_inference_file.npz')
    G_rirs = np.asarray(generator_rirs)
    true_rirs = np.asarray(true_rirs)
    input_rirs = np.asarray(input_rirs)
    np.savez(output_file,
             G_rirs = G_rirs,
             true_rirs = true_rirs,
             input_rirs =input_rirs,
             grid_ref = grid_ref,
             indices = indices)
    N = len(G_rirs)
    pbar2 = tqdm(range(N))

    for i in pbar2:
        hf = h5py.File(output_dir + f'/metrics_inference_{i}.h5', 'w')

        td_metrics_gen, time_intervals = time_metrics(G_rirs[i],
                                                       true_rirs[i],
                                                       fs=16000, t_0=0, t_end=300e-3)
        for k in td_metrics_gen.keys():
            hf['td_gan'+k]=td_metrics_gen[k]

        td_metrics_plwav, _ = time_metrics(input_rirs[i],
                                           true_rirs[i],
                                           fs=16000, t_0=0, t_end=300e-3)
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

        octave_band_metrics_gen, bands = octave_band_metrics(G_rirs[i],
                                                              true_rirs[i],
                                                              fs=16000)
        for k in octave_band_metrics_gen.keys():
            hf['octave_gan'+k]=octave_band_metrics_gen[k]

        octave_band_metrics_plwav, _ = octave_band_metrics(input_rirs[i],
                                                           true_rirs[i],
                                                           fs=16000)

        for k in octave_band_metrics_plwav.keys():
            hf['octave_plwav'+k]=octave_band_metrics_plwav[k]

        pbar2.set_description(f"Getting metrics for response : {i + 1}/{N}")

        hf['time_intervals'] = time_intervals
        hf['freq'] = freq
        hf['bands'] = bands
        hf.close()
        # np.savez(output_dir + '/x_axes.npz',
        #          time_intervals = time_intervals,
        #          freq = freq,
        #          bands = bands)
        # np.savez(output_dir + f'/metrics_gen_{i}.npz',
        #          td_metrics_gen = td_metrics_gen,
        #          td_metrics_plwav = td_metrics_plwav,
        #          fd_metrics_gen = fd_metrics_gen,
        #          fd_metrics_plwav = fd_metrics_plwav,
        #          octave_band_metrics_gen = octave_band_metrics_gen,
        #          octave_band_metrics_plwav = octave_band_metrics_plwav,
        #          time_intervals= time_intervals,
        #          freq = freq,
        #          bands = bands)


@click.command()
# options_metavar='<options>'
@click.option('--validation_dir', default='../validation_responses', type=str,
              help='Directory of validation data')
@click.option('--checkpoint_dir', default='./checkpoints_generator', type=str,
              help='Directory for saving model checkpoints')
@click.option('--output_dir', default='./generated_files', type=str,
              help='Directory for saving generator output')
@click.option('--config_file', default='HiFiGAN_config.yaml', type=str,
              help='Hyper-parameter and network architecture details stored in a .yaml file')
def main(validation_dir,
         checkpoint_dir,
         output_dir,
         config_file):
    print('Initializing Inference Process..')

    hp = config_from_yaml(config_file)

    torch.manual_seed(hp.seed)
    global device
    device = torch.device('cpu')

    inference(validation_dir, output_dir, checkpoint_dir, hp)


if __name__ == '__main__':
    main()


# name=  '/Users/xen/PhD Acoustics/Repositories/BWextension/hifi-extension/metrics_inference_1.h5'
# hf1 = h5py.File(name, 'r')
# for name in hf1:
#     print(name)
