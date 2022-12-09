import random

import h5py
import numpy as np
import torch
import torch.utils.data
from icecream import ic
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def read_npz(full_path):
    data = np.load(full_path, allow_pickle=True)
    aliased_ = data["aliased"]
    real_ = data["real"]
    return aliased_, real_


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def normalize_stft(stft, max=1.0):
    mag = torch.abs(stft)
    normmax = max * torch.max(mag)
    stft = stft / normmax
    return stft


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


mel_basis = {}
hann_window = {}


def mel_spectrogram(
        y, n_fft, num_mels,
        sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    print(y.shape)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


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
):
    global mel_basis, hann_window
    hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

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
        return_complex=not use_mel,
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
        spec_magnitude = normalize_stft(spec_magnitude, max=1.0)

    return spec_magnitude


def get_dataset_filelist(args):
    with open(args.train_file, "r", encoding="utf-8") as fi:
        training_files = [x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0]

    with open(args.valid_file, "r", encoding="utf-8") as fi:
        validation_files = [
            x.split("|")[0] for x in fi.read().split("\n") if len(x) > 0
        ]
    return training_files, validation_files


def validation_responses(file_name):
    # files = find_files(valid_path, extension)
    # assert len(file_list) > 0, "No {} files found".format(extension)
    with h5py.File(file_name, "r") as f:
        valid_true = f["pref"][:]
        valid_recon = f["prec"][:]
    f.close()
    return valid_true, valid_recon


class HiFiDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            training_files,
            segment_size,
            n_fft,
            num_mels,
            hop_size,
            win_size,
            sampling_rate,
            fmin,
            fmax,
            split=True,
            n_cache_reuse=1,
            shuffle=True,
            device=None,
            use_mel=False,
    ):
        self.training_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.training_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = False
        self.use_mel = use_mel

    def __getitem__(self, index):
        file_path = self.training_files[index]
        # file_path = os.path.join(self.input_dir, filename)
        if self._cache_ref_count == 0:
            # aliased_response, real_response = read_npz(file_path)
            # pdb.set_trace()
            with h5py.File(file_path, "r") as file:
                # Get the data
                # print("Keys: %s" % file.keys())
                aliased_response = file["prec"][:]
                real_response = file["pref"][:]

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

        assert aliased_response.size(1) == real_response.size(
            1
        ), "Inconsistent dataset length, unable to sampling"

        # if not self.fine_tuning:
        if self.split:
            if aliased_response.size(1) >= self.segment_size:
                aliased_response = aliased_response[:, : self.segment_size]
                real_response = real_response[:, : self.segment_size]
            else:
                aliased_response = torch.nn.functional.pad(
                    aliased_response,
                    (0, self.segment_size - aliased_response.size(1)),
                    "constant",
                )
                real_response = torch.nn.functional.pad(
                    real_response,
                    (0, self.segment_size - real_response.size(1)),
                    "constant",
                )
        # print(aliased_response.shape)
        # print(real_response.shape)
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
        return (
            aliased_response.squeeze(0),
            real_response.squeeze(0),
            file_path,
            real_spec.squeeze(),
        )

    def __len__(self):
        return len(self.training_files)


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            h5_path,
            segment_size,
            n_fft,
            num_mels,
            hop_size,
            win_size,
            sampling_rate,
            fmin,
            fmax,
            split=True,
            n_cache_reuse=1,
            shuffle=True,
            device=None,
            use_mel=False,
    ):
        random.seed(1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = False
        self.use_mel = use_mel
        self.h5_path = h5_path[0]
        ic(self.h5_path)
        # assert (self.h5_path.is_dir())
        with h5py.File(self.h5_path, "r") as temp_file:
            self.length = len(temp_file["pref"])
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

            # if not self.fine_tuning:
            # reconRIR = normalize(reconRIR) * 0.95
            self.cached_aliased_response = reconRIR

            # if not self.fine_tuning:
            # trueRIR = normalize(trueRIR) * 0.95
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

        assert reconRIR.size(1) == trueRIR.size(
            1
        ), "Inconsistent dataset length, unable to sampling"

        # if not self.fine_tuning:
        if self.split:
            if reconRIR.size(1) >= self.segment_size:
                reconRIR = reconRIR[:, : self.segment_size]
                trueRIR = trueRIR[:, : self.segment_size]
            else:
                reconRIR = torch.nn.functional.pad(
                    reconRIR, (0, self.segment_size - reconRIR.size(1)), "constant"
                )
                trueRIR = torch.nn.functional.pad(
                    trueRIR, (0, self.segment_size - trueRIR.size(1)), "constant"
                )
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

    def __len__(self):
        return self.length

# npz_files = find_files('.')
# trainset = MelDataset(npz_files, '', 15000, 512, 64, 256, 512, 16000, 20, 8000)
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
