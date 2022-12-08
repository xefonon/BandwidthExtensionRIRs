from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import torch
from librosa.util import normalize
import numpy as np
from src.models.SEGAN.SEGAN_plus import Generator
import wandb
import click
from src.models.SEGAN.SEGAN_utils import find_files
import h5py
from pathlib import Path
from tqdm.contrib import tzip

h = None
device = None


def validation_responses(file_name):
    # files = find_files(valid_path, extension)
    # assert len(file_list) > 0, "No {} files found".format(extension)
    with h5py.File(file_name, "r") as f:
        valid_true = f['ground_truth'][:]
        valid_recon = f['reconstructions'][:]
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
    generator = Generator(hp.in_channels).to(device)

    state_dict_g = load_checkpoint(checkpoint_dir, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = find_files(validation_dir, query='*.hdf5')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator.eval()
    # generator.remove_weight_norm()
    valid_true_all, valid_recon_all = validation_responses(filelist[0])  # shape: [709, 16384]

    true_rirs = []
    input_rirs = []
    generator_rirs = []
    with torch.no_grad():
        for valid_true, valid_recon in tzip(valid_true_all, valid_recon_all):
            true_rir = normalize(valid_true) * 0.95
            recon_rir = normalize(valid_recon) * 0.95

            true_rirs.append(true_rir)
            input_rirs.append(recon_rir)

            true_rir = torch.FloatTensor(true_rir)
            true_rir = true_rir.reshape((1, 1, true_rir.shape[0],)).to(device)

            recon_rir = torch.FloatTensor(recon_rir)
            recon_rir = recon_rir.reshape((1, 1, recon_rir.shape[0],)).to(device)

            y_generator, y_g_postnet = generator(recon_rir, with_postnet)
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
        print(output_file)


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

    wandb.init(config=config_file,
               allow_val_change=True,
               project='hifi_extension')

    hp = wandb.config
    torch.manual_seed(hp.seed)
    global device
    device = torch.device('cpu')

    inference(validation_dir, output_dir, checkpoint_dir, hp)


if __name__ == '__main__':
    run_inference_command()
