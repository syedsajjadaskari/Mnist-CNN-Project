
import os
import certifi
import urllib.request
import ssl

os.environ['SSL_CERT_FILE'] = certifi.where()

url = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

try:
    print(f"Attempting to download from {url}...")
    with urllib.request.urlopen(url) as response:
        print(f"Response code: {response.getcode()}")
        print("Download successful!")
except Exception as e:
    print(f"Download failed: {e}")
