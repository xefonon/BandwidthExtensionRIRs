import random

import h5py
import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from torch.utils import data

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def split_data(data_path_list, valid_ratio, test_ratio):
    num_files = len(data_path_list)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    random.shuffle(data_path_list)
    valid_files = data_path_list[:num_valid]
    test_files = data_path_list[num_valid:num_valid + num_test]
    train_files = data_path_list[num_valid + num_test:]
    train_size = len(train_files)
    return train_files, valid_files, test_files, train_size


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


def read_npz(full_path):
    data = np.load(full_path, allow_pickle=True)
    aliased_ = data['prec']
    real_ = data['pref']
    return aliased_, real_


def normalize_stft(stft, max=1.):
    mag = torch.abs(stft)
    normmax = max * torch.max(mag)
    stft = stft/normmax
    return stft


def validation_responses(file_name):
    with h5py.File(file_name, "r") as f:
        valid_true = f['rirs_ref'][:]
        valid_recon = f['rir_ridge'][:]
    f.close()
    return valid_true, valid_recon

mel_basis = {}
hann_window = {}

def spectrogram(
        y,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
        use_mel=False,
        return_log_spec = False
):
    global mel_basis, hann_window
    hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    if y.shape[-1] < 16384:
        y = torch.nn.functional.pad(
            y,
            (0, 16384 - y.shape[-1]),
            mode="constant",
        )
    y = torch.nn.functional.pad(
        y,
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="constant",
    )
    # y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=not use_mel
    )

    if use_mel:
        if fmax not in mel_basis:
            mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
            mel_basis[str(fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
        spec_magnitude = spectral_normalize_torch(spec)
    else:
        spec_magnitude = torch.abs(spec)
        if return_log_spec:
            spec_magnitude = torch.log(spec_magnitude + 1e-8)
        spec_magnitude = normalize_stft(spec_magnitude, max=1.0)

    return spec_magnitude


class RIRDataset(torch.utils.data.Dataset):
    def __init__(self,
                 training_files,
                 segment_size,
                 n_fft,
                 sampling_rate,
                 split=True,
                 num_mels=None,
                 hop_size=None,
                 win_size=None,
                 fmin=None,
                 fmax=None,
                 n_cache_reuse=1,
                 shuffle=True,
                 device=None,
                 use_spectrogram=False,
                 use_mel=False):

        self.num_mels = num_mels
        self.win_size = win_size
        self.hop_size = hop_size
        self.fmax = fmax
        self.fmin = fmin
        self.training_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.training_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = False
        self.use_mel = use_mel
        self.use_spectrogram = use_spectrogram

    def __getitem__(self, index):
        file_path = self.training_files[index]
        # file_path = os.path.join(self.input_dir, filename)
        if self._cache_ref_count == 0:
            aliased_response, real_response = read_npz(file_path)

            if not self.fine_tuning:
                aliased_response = normalize(aliased_response) * 0.95
            self.cached_aliased_response = aliased_response

            if not self.fine_tuning:
                real_response = normalize(real_response) * 0.95
            self.cached_real_response = real_response

            self._cache_ref_count = self.n_cache_reuse
        else:
            aliased_response = self.cached_aliased_response
            real_response = self.cached_real_response
            self._cache_ref_count -= 1

        aliased_response = torch.FloatTensor(aliased_response)
        aliased_response = aliased_response.unsqueeze(0)

        real_response = torch.FloatTensor(real_response)
        real_response = real_response.unsqueeze(0)

        assert aliased_response.size(1) == real_response.size(1), "Inconsistent dataset length, unable to initiate"

        if self.split:
            if aliased_response.size(1) >= self.segment_size:
                aliased_response = aliased_response[:, :self.segment_size]
                real_response = real_response[:, :self.segment_size]
            else:
                aliased_response = torch.nn.functional.pad(aliased_response,
                                                           (0, self.segment_size - aliased_response.size(1)),
                                                           'constant')
                real_response = torch.nn.functional.pad(real_response, (0, self.segment_size - real_response.size(1)),
                                                        'constant')

        if self.use_spectrogram:
            real_spec = spectrogram(
                real_response,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
                use_mel=self.use_mel,
            )

            return (aliased_response.squeeze(0), real_response.squeeze(0), file_path, real_spec.squeeze(0))
        else:
            return (aliased_response.squeeze(0), real_response.squeeze(0), file_path)

    def __len__(self):
        return len(self.training_files)

    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.
        Args:
            batch_size(int): batch size
        Returns:
            ref_batch: reference batch
        """
        ref_file_names = np.random.choice(self.training_files, batch_size)
        # ic(ref_file_names)
        # ic(self.training_files)
        ref_batch_real = []
        ref_batch_aliased = []
        for file in ref_file_names:
            aliased_response, real_response = read_npz(file)
            ref_batch_real.append(real_response)
            ref_batch_aliased.append(aliased_response)
        ref_batch_r = np.stack(ref_batch_real)
        ref_batch_a = np.stack(ref_batch_aliased)
        ref_batch_r = torch.from_numpy(ref_batch_r).type(torch.FloatTensor).unsqueeze(1)
        ref_batch_a = torch.from_numpy(ref_batch_a).type(torch.FloatTensor).unsqueeze(1)

        return torch.cat((ref_batch_a, ref_batch_r), dim=1)


class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 h5_path,
                 segment_size,
                 n_fft,
                 sampling_rate,
                 split=True,
                 num_mels=None,
                 hop_size=None,
                 win_size=None,
                 fmin=None,
                 fmax=None,
                 n_cache_reuse=1,
                 device=None,
                 use_spectrogram=False,
                 use_mel=False):
        self.h5_path = h5_path
        self.num_mels = num_mels
        self.win_size = win_size
        self.hop_size = hop_size
        self.fmax = fmax
        self.fmin = fmin
        random.seed(1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = False
        self.use_mel = use_mel
        self.use_spectrogram = use_spectrogram
        # ic(self.h5_path)
        with h5py.File(self.h5_path, 'r') as temp_file:
            # print(temp_file.keys())
            self.length = len(temp_file['rirs_ref'])
        temp_file.close()
        # if recursive:
        #     files = sorted(self.h5_path.glob('**/*.h5'))
        # else:
        #     files = sorted(self.h5_path.glob('*.h5'))
        # if len(files) < 1:
        #     raise RuntimeError('No hdf5 datasets found')

        self.data, self.labels = validation_responses(self.h5_path)

    def __getitem__(self, index):  # to enable indexing

        if self._cache_ref_count == 0:
            trueRIR = self.data[index]
            reconRIR = self.labels[index]

            if not self.fine_tuning:
                reconRIR = normalize(reconRIR) * 0.95
            self.cached_aliased_response = reconRIR

            if not self.fine_tuning:
                trueRIR = normalize(trueRIR) * 0.95
            self.cached_real_response = trueRIR

            self._cache_ref_count = self.n_cache_reuse
        else:
            reconRIR = self.cached_aliased_response
            trueRIR = self.cached_real_response
            self._cache_ref_count -= 1

        reconRIR = torch.FloatTensor(reconRIR)
        reconRIR = reconRIR.unsqueeze(0)

        trueRIR = torch.FloatTensor(trueRIR)
        trueRIR = trueRIR.unsqueeze(0)

        assert reconRIR.size(1) == trueRIR.size(1), "Inconsistent dataset length, unable to sampling"

        # if not self.fine_tuning:
        if self.split:
            if reconRIR.size(1) >= self.segment_size:
                reconRIR = reconRIR[:, :self.segment_size]
                trueRIR = trueRIR[:, :self.segment_size]
            else:
                reconRIR = torch.nn.functional.pad(reconRIR,
                                                   (0, self.segment_size - reconRIR.size(1)),
                                                   'constant')
                trueRIR = torch.nn.functional.pad(trueRIR, (0, self.segment_size - trueRIR.size(1)),
                                                  'constant')

        if self.use_spectrogram:
            real_spec = spectrogram(
                trueRIR,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
                use_mel=self.use_mel,
            )

            return (reconRIR.squeeze(0), trueRIR.squeeze(0), real_spec.squeeze())
        else:
            return (reconRIR.squeeze(0), trueRIR.squeeze(0))

    def __len__(self):
        return self.length

# npz_files = find_files('.')
# trainset = ValidationDataset(npz_files, '',)
# from torch.utils.data import DistributedSampler, DataLoader
#
# train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
#                           sampler=None,
#                           batch_size=1,
#                           pin_memory=True,
#                           drop_last=True)
#
# for i, batch in enumerate(train_loader):
#     x, y, filename, mel = batch
#     print("x shape: ", x.shape)
#     print("y shape: ", y.shape)
#     print("mel shape: ", mel.shape)
#     print("filename : ", filename)
# import matplotlib.pyplot as plt
# gt, rec = validation_responses('/Users/xen/PhD Acoustics/Repositories/BWextension/validation_responses/IEC_reconstructions.hdf5')
# spec = spectrogram(torch.from_numpy(gt[0, :4096]), 2048, None, 16000, 512, 2048, 0, 8e3)

# plt.imshow(spec); plt.show()