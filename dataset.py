import os
import tarfile
import urllib.request
import numpy as np
from PIL import Image

if not os.path.exists("stl10"):
    os.makedirs("stl10")

url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
output_path = os.path.join("stl10", "stl10_binary.tar.gz")

if not os.path.exists(output_path):
    print("Downloading STL-10 dataset...")
    urllib.request.urlretrieve(url, output_path)
    print("Download complete!")
else:
    print("Dataset already downloaded.")

with tarfile.open(output_path, "r:gz") as tar:
    tar.extractall(path="stl10/")

print("Dataset extracted!")

def read_images(bin_file, out_dir):
    # Each image is 96x96x3 uint8
    with open(bin_file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        images = data.reshape(-1, 3, 96, 96).transpose(0,2,3,1)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for i, img in enumerate(images):
        im = Image.fromarray(img)
        im.save(os.path.join(out_dir, f"{i}.png"))

# Convert training and test images
read_images("stl10/stl10_binary/train_X.bin", "stl10/images/train")
read_images("stl10/stl10_binary/test_X.bin", "stl10/images/test")

print("All images saved as PNGs in 'stl10/images/train' and 'stl10/images/test'")
