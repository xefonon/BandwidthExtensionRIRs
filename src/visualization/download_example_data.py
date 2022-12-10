import urllib.request
import pathlib
import os

hifi_checkpoint_path = pathlib.Path("..",  "models", "HiFiGAN", "checkpoints_generator")
segan_checkpoint_path = pathlib.Path("..", "models", "SEGAN", "checkpoints_generator")
csgm_checkpoint_path = pathlib.Path("..", "models", "CSGM", "Generator_checkpoints")
test_data_path = pathlib.Path("..", "..", "data", "Inference Data")

urls = [
  'URLHERE',
  'URLHERE',
  'URLHERE',
  'URLHERE'
]

Hifi_weights_url = urls[0]
filename = Hifi_weights_url.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(Hifi_weights_url, os.path.join(str(hifi_checkpoint_path), filename))
print("Done.")

segan_weights_url = urls[1]
filename = segan_weights_url.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(segan_weights_url, os.path.join(str(segan_checkpoint_path), filename))
print("Done.")

csgm_weights_url = urls[2]
filename = csgm_weights_url.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(csgm_weights_url, os.path.join(str(csgm_checkpoint_path), filename))
print("Done.")

input_data = urls[3]
filename = input_data.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(input_data, os.path.join(str(test_data_path), filename))
print("Done.")

target_data = urls[4]
filename = target_data.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(target_data, os.path.join(str(test_data_path), filename))
print("Done.")

