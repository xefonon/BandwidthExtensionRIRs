import torch
from src.models.CSGM.CSGM_models import load_generator, CSGM, AdaptiveCSGM
import click
import datetime
import yaml
from src.models.CSGM.utils import config_from_yaml, plot_rir, plot_frf, normalize
from src.models.CSGM.utils import find_files
import matplotlib.pyplot as plt
import numpy as np
from src.validation_responses.evaluation_metrics import time_metrics, freq_metrics, octave_band_metrics
import os
from tqdm.auto import tqdm
import h5py


def select_responses(ref_responses, rec_responses, grid_ref, return_indices=True):
    hsph = 0.628
    r = np.linalg.norm(grid_ref - np.array([[0., 0., hsph]]).T, axis=0)

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


@click.command()
@click.option('--use_wandb', is_flag=True,
              help='Use weights and biases to monitor training')
@click.option('--config_file', default='HiFiGAN_config.yaml', type=str,
              help='Configuration (.yaml) file including hyperparameters for training')
@click.option('--checkpoint_dir', default='Generator_checkpoints', type=str,
              help='Directory of trained generator model')
@click.option('--adaptive_gan', is_flag=True,
              help='Use adaptive GAN after CSGM (latent) optimisation')
def reconstruct(use_wandb, config_file, checkpoint_dir, adaptive_gan):
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    if use_wandb:
        import wandb
        date_ = datetime.date.today().strftime('%m-%d')
        time_ = datetime.datetime.now().strftime("%H:%M")
        print("Using Weights and Biases to track training!")
        wandb.login()
        config_dict = yaml.load(open(config_file), Loader=yaml.FullLoader)
        run = wandb.init(project='CSGM_inference',
                         name=config_dict['model_name']['value'] + '_' + date_ + '_' + time_,
                         config=config_file)
        config = wandb.config
    else:
        config = config_from_yaml(config_file)
        run = None

    G = load_generator(checkpoint_dir, config)

    responses_true, responses_aliased = find_validation_data(config.validation_dir, query="*.hdf5")

    response_true = (normalize(torch.from_numpy(responses_true[562])) * 0.95).unsqueeze(0)
    response_aliased = (normalize(torch.from_numpy(responses_aliased[562])) * 0.95).unsqueeze(0)
    response_CSGM, z_CSGM, losses_CSGM = CSGM(response_aliased, G, config, run)

    if use_wandb:
        ax1 = plot_frf(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                       y_pred=response_aliased[0, 0, :].cpu().detach().numpy(),
                       title='Plane wave reconstruction response')
        ax2 = plot_frf(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                       y_pred=response_CSGM[0, 0, :].cpu().detach().numpy(),
                       title='CSGAN response')

        run.log({"CSGM/FRF_plane_wave": wandb.Image(ax1)})
        run.log({"CSGM/FRF_CSGAN": wandb.Image(ax2)})
        plt.close('all')

        ax3 = plot_rir(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                       y_pred=response_aliased[0, 0, :].cpu().detach().numpy(),
                       title='Plane wave reconstruction RIR',
                       y_pred_lab='Plane Wave Reconstruction (Input)')
        ax4 = plot_rir(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                       y_pred=response_CSGM[0, 0, :].cpu().detach().numpy(),
                       title='CSGM output RIR',
                       y_pred_lab='GAN Reconstruction RIR (Input)')
        run.log({"CSGM/RIR_plane_wave": wandb.Image(ax3)})
        run.log({"CSGM/RIR_CSGAN": wandb.Image(ax4)})
        # wandb.log(log_dict)
    if adaptive_gan:
        response_adapt_CSGM, losses_adapt_CSGM = AdaptiveCSGM(response_aliased, G, z_CSGM, config, run)

        if use_wandb:
            # log_dict = {'Adaptive_CSGM/' + key: losses_adapt_CSGM[key] for key in losses_adapt_CSGM.keys()}
            # log_dict.update({'steps': steps, 'pearson_corr_coeff': pearson})
            cossim_val = cos_sim(response_true.squeeze(0).cpu(), response_adapt_CSGM.squeeze(0).cpu()).item()

            ax1 = plot_frf(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                           y_pred=response_aliased[0, 0, :].cpu().detach().numpy(),
                           title='Plane wave reconstruction response')
            ax2 = plot_frf(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                           y_pred=response_adapt_CSGM[0, 0, :].cpu().detach().numpy(),
                           title='CSGAN response')
            run.log({"Adaptive_CSGM/FRF_plane_wave": wandb.Image(ax1)})
            run.log({"Adaptive_CSGM/FRF_CSGAN": wandb.Image(ax2)})
            plt.close('all')

            ax3 = plot_rir(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                           y_pred=response_aliased[0, 0, :].cpu().detach().numpy(),
                           title='Plane wave reconstruction RIR',
                           y_pred_lab='Plane Wave Reconstruction (Input)')
            ax4 = plot_rir(y_truth=response_true[0, 0, :].cpu().detach().numpy(),
                           y_pred=response_adapt_CSGM[0, 0, :].cpu().detach().numpy(),
                           title='CSGM output RIR',
                           y_pred_lab='GAN Reconstruction RIR (Input)')
            run.log({"Adaptive_CSGM/RIR_plane_wave": wandb.Image(ax3)})
            run.log({"Adaptive_CSGM/RIR_CSGAN": wandb.Image(ax4)})
            run.log({"cos_sim": cossim_val})
            # wandb.log(log_dict)
            run.finish()


@click.command()
@click.option('--config_file', default='HiFiGAN_config.yaml', type=str,
              help='Configuration (.yaml) file including hyperparameters for training')
@click.option('--checkpoint_dir', default='Generator_checkpoints', type=str,
              help='Directory of trained generator model')
@click.option('--adaptive_gan', is_flag=True,
              help='Use adaptive GAN after CSGM (latent) optimisation')
@click.option('--get_metrics', is_flag=True,
              help='Use adaptive GAN after CSGM (latent) optimisation')
@click.option('--inference_dir', default='inference_data', type=str,
              help='Directory of inferred data')
def inference_command(config_file, checkpoint_dir, adaptive_gan, inference_dir, get_metrics):
    return inference(config_file, checkpoint_dir, adaptive_gan, inference_dir, get_metrics)

def inference(config_file, checkpoint_dir, adaptive_gan, inference_dir, get_metrics,
              validation_dir):
    config = config_from_yaml(config_file)
    run = None

    G = load_generator(checkpoint_dir, config)

    valid_true_all, valid_recon_all, grid_ref = find_validation_data(validation_dir, return_grid=True)

    indices = select_responses(valid_true_all, valid_recon_all, grid_ref, return_indices=True)
    valid_true_all = valid_true_all[indices]
    valid_recon_all = valid_recon_all[indices]
    total_responses = len(valid_true_all)

    print(80*'-')
    print(f"Reconstructing a total of {total_responses} RIRs.")
    print(f"Will perform csgm optimisation for {config.n_z_init} latent variable initialisations.")
    print(80*'-')

    grid_ref = grid_ref[:, indices]
    # total_responses = 1
    inference_data = {}
    gt = []
    rec = []
    csgm = []
    z = []
    adaptcsgm = []
    pbar = tqdm(range(total_responses))
    pbar.set_description(f"Performing inference for response : { 1}/{total_responses}")

    for i in pbar:
        response_true = (normalize(torch.from_numpy(valid_true_all[i])) * 0.95).unsqueeze(0)
        response_aliased = (normalize(torch.from_numpy(valid_recon_all[i])) * 0.95).unsqueeze(0)
        response_CSGM, z_CSGM, losses_CSGM = CSGM(response_aliased, G, config, run)

        gt.append(response_true.squeeze(0).squeeze(0).data.numpy())
        rec.append(response_aliased.squeeze(0).squeeze(0).data.numpy())
        csgm.append(response_CSGM.squeeze(0).squeeze(0).data.cpu().numpy())
        z.append(z_CSGM.squeeze(0).squeeze(0).data.cpu().numpy())
        if adaptive_gan:
            response_adapt_CSGM, losses_adapt_CSGM = AdaptiveCSGM(response_aliased, G, z_CSGM, config, run)
            adaptcsgm.append(response_adapt_CSGM.squeeze(0).squeeze(0).data.cpu().numpy())
        pbar.set_description(f"Performing inference for response : {i + 1}/{total_responses}")
    inference_data['responses_true'] = np.array(gt)
    inference_data['responses_aliased'] = np.array(rec)
    inference_data['responses_CSGM'] = np.array(csgm)
    inference_data['Z_CSGM'] = np.array(z)
    inference_data['responses_adCSGM'] = np.array(adaptcsgm)
    inference_data['grid_ref'] = np.array(grid_ref)
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    if get_metrics:
        N = len(inference_data['responses_true'])
        with tqdm(range(N), position=0, leave=True, ascii=True) as iterator:

            for i in iterator:
                hf = h5py.File(inference_dir + f'/metrics_inference_{i}.h5', 'w')

                td_metrics_gen, time_intervals = time_metrics(inference_data['responses_adCSGM'][i],
                                                              inference_data['responses_true'][i],
                                                              fs=16000, t_0=0, t_end=300e-3)
                for k in td_metrics_gen.keys():
                    hf['td_gan' + k] = td_metrics_gen[k]

                td_metrics_plwav, _ = time_metrics(inference_data['responses_aliased'][i],
                                                   inference_data['responses_true'][i],
                                                   fs=16000, t_0=0, t_end=300e-3)
                for k in td_metrics_plwav.keys():
                    hf['td_plwav' + k] = td_metrics_plwav[k]

                fd_metrics_gen, freq = freq_metrics(inference_data['responses_adCSGM'][i],
                                                    inference_data['responses_true'][i],
                                                    fs=16000)
                for k in fd_metrics_gen.keys():
                    hf['fd_gan' + k] = fd_metrics_gen[k]

                fd_metrics_plwav, _ = freq_metrics(inference_data['responses_aliased'][i],
                                                   inference_data['responses_true'][i],
                                                   fs=16000, index=i)
                for k in fd_metrics_plwav.keys():
                    hf['fd_plwav' + k] = fd_metrics_plwav[k]

                octave_band_metrics_gen, bands = octave_band_metrics(inference_data['responses_adCSGM'][i],
                                                                     inference_data['responses_true'][i],
                                                                     fs=16000)
                for k in octave_band_metrics_gen.keys():
                    hf['octave_gan' + k] = octave_band_metrics_gen[k]

                octave_band_metrics_plwav, _ = octave_band_metrics(inference_data['responses_aliased'][i],
                                                                   inference_data['responses_true'][i],
                                                                   fs=16000)

                for k in octave_band_metrics_plwav.keys():
                    hf['octave_plwav' + k] = octave_band_metrics_plwav[k]

                iterator.set_postfix_str(f"Getting metrics for response : {i + 1}/{N}")

                hf['time_intervals'] = time_intervals
                hf['freq'] = freq
                hf['bands'] = bands
                hf.close()

    output_file = os.path.join(
        inference_dir, 'inference_data.npz')

    np.savez(output_file,
             responses_true=inference_data['responses_true'],
             responses_aliased=inference_data['responses_aliased'],
             responses_CSGM=inference_data['responses_CSGM'],
             Z_CSGM=inference_data['Z_CSGM'],
             responses_adCSGM=inference_data['responses_adCSGM'],
             grid_ref=inference_data['grid_ref'])


    print(80 * '=')
    print("Inference RIRs saved in path: ", output_file)
    print(80 * '=')

if __name__ == '__main__':
    # reconstruct()
    inference_command()
