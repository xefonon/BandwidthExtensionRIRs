from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import torch
from librosa.util import normalize
import numpy as np
from src.models.SEGAN.SEGAN_plus import Generator
import click
from src.models.SEGAN.SEGAN_utils import find_files, config_from_yaml
import h5py
from pathlib import Path
from tqdm.contrib import tzip
from tqdm.autonotebook import tqdm

h = None
device = None

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

def select_responses(ref_responses, rec_responses, grid_ref, return_indices=True):
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
        prev_radius = radii[i - 1] if i > 0 else 1e-18
        indx = np.squeeze(
            np.argwhere((r > prev_radius) & (r <= radius)))
        selected_indx = np.random.choice(indx, 1, replace=False) if len(indx) > 1 else indx
        selected_ref_responses.append(ref_responses[selected_indx])
        selected_rec_responses.append(rec_responses[selected_indx])
        selected_grid.append(grid_ref[:, selected_indx])
        indices.append(selected_indx)
    if return_indices:
        return np.squeeze(indices)
    else:
        return np.array(selected_ref_responses), np.array(selected_rec_responses), np.array(selected_grid)

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


def inference(validation_dir, output_dir, checkpoint_dir, hp):
    generator = Generator(input_size = 1, kernel_size= hp.kernel_size).to(device)
    checkpoint = scan_checkpoint(checkpoint_dir, 'g_')
    state_dict_g = load_checkpoint(checkpoint, device)
    # state_dict_g = load_checkpoint(checkpoint_dir, device)
    generator.load_state_dict(state_dict_g['generator'])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.eval()

    valid_true_all, valid_recon_all, grid_ref = find_validation_data(validation_dir,
                                                                     return_grid=True)  # shape: [709, 16384]

    indices = select_responses(valid_true_all, valid_recon_all, grid_ref, return_indices=True)
    valid_true_all = valid_true_all[indices]
    valid_recon_all = valid_recon_all[indices]
    grid_ref = grid_ref[:, indices]
    true_rirs = []
    input_rirs = []
    generator_rirs = []
    pbar = tqdm(zip(valid_true_all, valid_recon_all))
    with torch.no_grad():
        for valid_true, valid_recon in pbar:
            true_rir = normalize(valid_true) #* 0.95
            recon_rir = normalize(valid_recon) #* 0.95

            true_rirs.append(true_rir)
            input_rirs.append(recon_rir)

            true_rir = torch.FloatTensor(true_rir)
            true_rir = true_rir.reshape((1, 1, true_rir.shape[0],)).to(device)

            recon_rir = torch.FloatTensor(recon_rir)
            recon_rir = recon_rir.reshape((1, 1, recon_rir.shape[0],)).to(device)
            y_generator = generator(recon_rir)
            G_rir = y_generator.reshape((y_generator.shape[2],))
            # audio = audio * MAX_WAV_VALUE
            # G_rir = G_rir.cpu().numpy().astype('int16')
            G_rir = G_rir.cpu().numpy()
            generator_rirs.append(G_rir)

        output_file = os.path.join(
            output_dir, 'generator_inference.npz')
        np.savez(output_file,
                 G_rirs=np.asarray(generator_rirs),
                 true_rirs=np.asarray(true_rirs),
                 input_rirs=np.asarray(input_rirs))
        print(80*'=')
        print("Inference RIRs saved in path: ", output_file)
        print(80*'=')

@click.command()
# options_metavar='<options>'
@click.option('--validation_dir', default='./validation_responses', type=str,
              help='Directory of validation data')
@click.option('--checkpoint_dir', default='./checkpoints_generator/g_00680000', type=str,
              help='Directory for saving model checkpoints')
@click.option('--output_dir', default='./generated_files', type=str,
              help='Directory for saving generator output')
@click.option('--config_file', default='HiFiGAN_config.yaml', type=str,
              help='Hyper-parameter and network architecture details stored in a .yaml file')
def run_inference_command(validation_dir,
                          checkpoint_dir,
                          output_dir,
                          config_file):
    return run_inference(validation_dir,
                         checkpoint_dir,
                         output_dir,
                         config_file)


def run_inference(validation_dir,
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
    run_inference_command()
