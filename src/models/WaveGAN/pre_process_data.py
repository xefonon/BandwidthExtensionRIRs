import numpy as np
import os
from librosa import load, resample
import time
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from tqdm import tqdm
# filename = "/Users/xen/PhD Acoustics/Repositories/BWextension/WaveGAN/RIRdata/rir_011.h5"

def read_h5_data(filename):
    rirs = {}
    with h5py.File(filename, "r") as f:
        # List all groups
        for key in f.keys():
            # print('key: ', key)
            for kkey in f[key].keys():
                if kkey == 'impulse_response':
                    rirs[key] = np.array(f[key][kkey])
    return rirs


meshrir_indices = np.arange(0, 32, 4)
rootdir = 'RIRdata/BUTReverbDB'
numpy_data_root = './NewNPdata'
# new_data_folder = os.path.join(rootdir, numpy_data_root)
new_data_folder = numpy_data_root
Path(new_data_folder).mkdir(parents=True, exist_ok=True)

for root, subdirs, files in os.walk(rootdir):

    # print(f"root: {root}, subdirs: {subdirs}, files {files}", end = "")

    for file in tqdm(files, desc = 'Processing files...'):
        # print('found %s' % os.path.join(root, file))
        if file.split('.')[-1] == 'wav':
            new_folder = root.split('/')[-1]
            rir, fs = load(os.path.join(root, file), sr = 16000)
            path_np = os.path.join(new_data_folder, new_folder)
            Path(path_np).mkdir(parents=True, exist_ok=True)
            np.save(os.path.join(path_np,file.split('.wav')[0]), rir)
            # plt.plot(rir)
            # plt.title(file)
            # plt.show()
            # time.sleep(0.5)
            # plt.close()
        if file.split('.')[-1] == 'npy':
            if file.split('_')[0] == 'ir':
                meshrir_folder = 'MeshRIRset'
                path_meshrir = os.path.join(numpy_data_root, meshrir_folder)
                Path(path_meshrir).mkdir(parents=True, exist_ok=True)
                rirs = np.load(os.path.join(root, file))
                rirs = rirs[meshrir_indices]

                for ii, rir in enumerate(rirs):
                    path_np = os.path.join(path_meshrir, "source_pos_{}".format(ii))
                    Path(path_np).mkdir(parents=True, exist_ok=True)
                    rir = resample(rir, 48000, 16000)
                    np.save(os.path.join(path_np,file.split('.npy')[0]), rir)
        if file.split('.')[-1] == 'h5':
            rirs = read_h5_data(os.path.join(root, file))
            h5folder = os.path.join(new_data_folder, '011_data_SAMUEL')
            Path(h5folder).mkdir(parents=True, exist_ok=True)
            for key in rirs.keys():
                new_folder = os.path.join(h5folder, key)
                Path(new_folder).mkdir(parents=True, exist_ok=True)
                for ii, rir in enumerate(rirs[key]):
                    rir = resample(rir, 44100, 16000)
                    np.save(os.path.join(new_folder,f'ir_{ii}'), rir)





