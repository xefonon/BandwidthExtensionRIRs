import os
import random
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
import torch


def get_npy_filename_list(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                all_files.append(os.path.join(root,file))
    return all_files

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

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def read_npz(full_path):
    rir = np.load(full_path)
    return rir

def get_dataset_filelist(args):
    with open(args.train_file, 'r', encoding='utf-8') as fi:
        training_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(args.valid_file, 'r', encoding='utf-8') as fi:
        validation_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

# class resampleRIR(object):
#     """Resample the RIR in to a given sample rate.
#
#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#
#     def __init__(self, new_sample_rate, old_sample_rate):
#         assert isinstance(new_sample_rate, (int, tuple))
#         self.new_sample_rate = new_sample_rate
#         self.old_sample_rate = old_sample_rate
#
#     def __call__(self, rir):
#         resampled_rir = F.resample(rir, self.old_sample_rate, self.new_sample_rate, lowpass_filter_width=128)
#         return resampled_rir

class WaveGANDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, sampling_rate = 16000, n_cache_reuse=1, normalize = True,
                 device=None, truncate = False, pad_lr = True, shuffle = False, data_sampling_rate = 16000):
        self.training_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.training_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.truncate = truncate
        self.normalize = normalize
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.pad_lr = pad_lr
        self.device = device
        self.fs = data_sampling_rate

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        file_path = self.training_files[index]
        # file_path = os.path.join(self.input_dir, filename)
        if self._cache_ref_count == 0:
            rir = read_npz(file_path)

            if self.normalize:
                rir = normalize(rir) * 0.95
            self.cached_rir = rir
            self._cache_ref_count = self.n_cache_reuse
        else:
            rir = self.cached_rir
            self._cache_ref_count -= 1

        rir = torch.FloatTensor(rir)
        max_i = torch.argmax(torch.abs(rir))
        rir = rir.unsqueeze(0)

        if self.truncate:
            if rir.size(1) >= self.segment_size:
                rir = rir[:, :self.segment_size]
            else:
                rir = torch.nn.functional.pad(rir,
                                              (0, self.segment_size - rir.size(1)),
                                              'constant')
        if self.pad_lr:
            if rir.size(1) >= self.segment_size:
                rir = rir[:, :self.segment_size]
            max_i = torch.max(max_i, max_i - 100)
            if max_i == 0:
                max_i += 1
            shift = torch.randint(low = 0, high = max_i, size = (1,))
            if shift > self.segment_size//2:
                shift = np.random.randint(1, 100)
            elif shift == 0:
                shift = np.random.randint(1, 100)
            criteria = torch.bernoulli(torch.tensor(0.5))
            if criteria:

                n_padding = self.segment_size - rir.size(1)
                if n_padding >  rir.size(1):
                    rir = torch.nn.functional.pad(rir,
                                                  (0, self.segment_size - rir.size(1)),
                                                  'constant')
                else:
                    rir = torch.nn.functional.pad(rir,
                                                  (0, self.segment_size - rir.size(1)),
                                                  'reflect')

                rir = torch.nn.functional.pad(rir,
                                              (shift, 0),
                                              'constant')
                rir = rir[:, :-shift]
            else:
                n_padding = self.segment_size - rir.size(1)
                if n_padding >  rir.size(1):
                    rir = torch.nn.functional.pad(rir,
                                                  (0, self.segment_size - rir.size(1)),
                                                  'constant')
                else:
                    rir = torch.nn.functional.pad(rir,
                                                  (0, self.segment_size - rir.size(1)),
                                                  'reflect')

                rir = rir[:, shift:]
                rir = torch.nn.functional.pad(rir,
                                              (shift, 0),
                                              'constant')
        return (rir, file_path)

    def __len__(self):
        return len(self.training_files)
