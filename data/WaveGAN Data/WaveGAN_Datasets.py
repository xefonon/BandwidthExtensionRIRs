import urllib.request
import pathlib
import os
import tarfile
import zipfile
import librosa
import time

import numpy as np
from scipy.io import loadmat

datadir = pathlib.Path('external')
rawdir = pathlib.Path('raw')
savedir = pathlib.Path('processed')

datadir.mkdir(parents=True, exist_ok=True)
rawdir.mkdir(parents=True, exist_ok=True)
savedir.mkdir(parents=True, exist_ok=True)

""" DTU dataset """
start = time.time()
print(80*'-*')
print("Downloading DTU dataset...")
# User must have an IEEE account
# You must be an IEEE DataPort subscriber:
data_URL1 = 'https://ieee-dataport.s3.amazonaws.com/open/67653/DTU_RIR_DOA_dataset.zip?response-content-disposition=attachment%3B%20filename%3D%22DTU_RIR_DOA_dataset.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20221207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221207T121552Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=3600&X-Amz-Signature=baa8e7f188f2b1aca56f5f957717cdd97f65f36f64280f468ca03ac2d2fab2e2'
output_path1 = os.path.join(str(datadir.absolute()), 'dtu_data.zip')
urllib.request.urlretrieve(data_URL1, output_path1)

end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))   # '04h13m06s'
print(f"Downloaded DTU Dataset, took {total_time}")
print(80*'-*')

print("Extracting DTU Dataset and pre-processing...")
start = time.time()
# extract DTU files (1 .mat file)
with zipfile.ZipFile(output_path1, 'r') as zip_ref:
    zip_ref.extractall(str(rawdir.absolute()))

# load raw DTU files
DTU_filename = os.path.join(str(rawdir.absolute()), 'DTU_RIR_DOA_dataset')
DTU_filename = os.path.join(DTU_filename, 'RIRs_DTU_3ch_DOA.mat')
DTU_raw_file = loadmat(DTU_filename, simplify_cells = True)

# append raw DTU files into list
Fs = DTU_raw_file['Fs']
new_Fs = 16000
DTU_RIRs_dict = DTU_raw_file['RIRs']
DTU_RIRs = []
for key in DTU_RIRs_dict.keys():
    for key2 in DTU_RIRs_dict[key].keys():
        if np.logical_and(key != 'noise', key != 'param'):
            DTU_RIRs.append(librosa.resample(DTU_RIRs_dict[key][key2],Fs, new_Fs))

end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
print(80*'-*')
print(f"Processed DTU Dataset, took {total_time}")
print(80*'-*')

""" Mesh RIR dataset"""
# https://www.sh01.org/MeshRIR/

start = time.time()
print(80*'-*')
print("Downloading MeshRIR dataset...")

data_URL2 = 'https://zenodo.org/record/5500451/files/S32-M441_npy.zip?download=1'
output_path2 = os.path.join(str(datadir.absolute()), 'MeshRIR_data.zip')
urllib.request.urlretrieve(data_URL2,  output_path2)

end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
print(80*'-*')
print(f"Downloaded MeshRIR dataset, took {total_time}.")
print(80*'-*')

print(80*'-*')
print("Pre-processing MeshRIR dataset...")
start = time.time()

# extract MeshRIR files (.npy files)
with zipfile.ZipFile(output_path2, 'r') as zip_ref:
    zip_ref.extractall(str(rawdir.absolute()))

newpath_2 = os.path.join(str(rawdir.absolute()), 'S32-M441_npy')
MeshRIRfilepaths = list(pathlib.Path(newpath_2).glob('**/*ir_*.npy'))

MeshRIR_source_positions = np.arange(0, 32, 3) # chosen 11 at random
MeshRIRfiles = list(np.load(MeshRIRfilepaths[0])[MeshRIR_source_positions])
for filepath in MeshRIRfilepaths[1:]:
    temp = np.load(filepath)
    MeshRIRfiles += list(np.load(filepath)[MeshRIR_source_positions])

# resample MeshRIR dataset
MeshRIRfiles_resampled = []
for rir in MeshRIRfiles:
    MeshRIRfiles_resampled.append(librosa.resample(rir, Fs, new_Fs))

# remove some silence at the start (120 ms):

MeshRIRfiles_resampled = np.array(MeshRIRfiles_resampled)[:, int(new_Fs*0.12):]
end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
print(80*'-*')
print(f"Processed MeshRIR dataset, took {total_time}.")
print(80*'-*')

""" Ace-Corpus dataset 1"""
start = time.time()
print(80*'-*')
print("Downloading AceCorpus dataset 1...")

# maybe you need to register for an account first (see details below)
# http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
data_URL3 = 'https://acecorpus.ee.ic.ac.uk/wp-content/uploads/edd/ACE_Corpus_RIRN_Crucif.tbz2'
output_path3 =  os.path.join(str(datadir.absolute()), 'AceCorpus1.tbz2')
urllib.request.urlretrieve(data_URL3, output_path3)

end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
print(80*'-*')
print(f"Downloaded AceCorpus dataset 1, took {total_time}.")
print(80*'-*')


print(80*'-*')
print("Pre-processing AceCorpus dataset 1...")
start = time.time()

tar = tarfile.open(output_path3, "r:bz2")
tar.extractall(str(rawdir.absolute()))
tar.close()

extracted_path_ace1 = os.path.join(str(rawdir.absolute()),'Crucif' )
Ace_wavfilepaths1 = list(pathlib.Path(extracted_path_ace1).glob('**/*RIR.wav'))

Ace_rirs = []
for filepath in Ace_wavfilepaths1:
    temp, Fs_wav = librosa.load(filepath, sr = 16000, mono = False)
    Ace_rirs += list(temp)

end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
print(80*'-*')
print(f"Processed AceCorpus dataset 1, took {total_time}.")
print(80*'-*')

""" Ace-Corpus dataset 2"""
# maybe you need to register for an account first (see details below)
# http://www.ee.ic.ac.uk/naylor/ACEweb/index.html
start = time.time()
print(80*'-*')
print("Downloading AceCorpus dataset 2...")

data_URL3 = 'https://acecorpus.ee.ic.ac.uk/wp-content/uploads/edd/ACE_Corpus_RIRN_Lin8Ch.tbz2'
output_path4 =  os.path.join(str(datadir.absolute()), 'AceCorpus2.tbz2')
urllib.request.urlretrieve(data_URL3, output_path4)
end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))

print(80*'-*')
print(f"Downloaded AceCorpus dataset 2, took {total_time}.")
print(80*'-*')

print(80*'-*')
print("Pre-processing AceCorpus dataset 2...")
start = time.time()

tar = tarfile.open(output_path4, "r:bz2")
tar.extractall(str(rawdir.absolute()))
tar.close()

extracted_path_ace2 = os.path.join(str(rawdir.absolute()),'Lin8Ch' )
Ace_wavfilepaths2 = list(pathlib.Path(extracted_path_ace2).glob('**/*RIR.wav'))

for filepath in Ace_wavfilepaths2:
    temp, Fs_wav = librosa.load(filepath, sr = 16000, mono = False)
    Ace_rirs += list(temp)

end = time.time()
elapsed = end-start
total_time = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed))
print(80*'-*')
print(f"Processed AceCorpus dataset 2, took {total_time}.")
print(80*'-*')

""" Save processed rirs"""
# truncate to N = 16384 samples
N = 16384
for i, rir in enumerate(DTU_RIRs):
    np.save(os.path.join(str(savedir.absolute()), f"DTU_rir_{i}.npy"), rir[:N])

for i, rir in enumerate(MeshRIRfiles_resampled):
    np.save(os.path.join(str(savedir.absolute()), f"MeshRIR_{i}.npy"), rir[:N])

for i, rir in enumerate(Ace_rirs):
    np.save(os.path.join(str(savedir.absolute()), f"AceCorpus_{i}.npy"), rir[:N])
