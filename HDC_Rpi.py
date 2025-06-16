
# Import libraries
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchhd
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
import psutil
import time
from torchhd.models import Centroid
from torchhd import embeddings
from torchvision.datasets.utils import download_and_extract_archive

# Use the CPU
device = torch.device("cpu")
print("Using CPU device")

# Hyperparameters
DIMENSIONS = 5000  # Reduced dimensionality for Raspberry Pi
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1  # Batch size of 1 for limited CPU resources

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.normalize(sample_hv)  # Updated to avoid deprecation warning
        return sample_hv

# Prepare the dataset manually to avoid HTTP errors
def download_mnist():
    urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    ]
    for url in urls:
        download_and_extract_archive(url, download_root="./data/MNIST/raw")

download_mnist()

# Load the MNIST dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_ds = MNIST(root="./data", train=True, download=False, transform=transform)
test_ds = MNIST(root="./data", train=False, download=False, transform=transform)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Encoder and Model
encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS).to(device)
num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes).to(device)

# Function to measure memory usage
def print_memory_usage():
    memory_info = psutil.virtual_memory()
    print(f"Memory Usage: {memory_info.used / 1024**2:.2f} MB / {memory_info.total / 1024**2:.2f} MB")

# Training Function
def train():
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)
            samples_hv = encode(samples)
            model.add(samples_hv, labels)

# Testing Function
def test():
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    with torch.no_grad():
        model.normalize()
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), labels)
    acc = accuracy.compute().item() * 100
    print(f"Testing accuracy: {acc:.3f}%")

# Measure performance (time)
def measure_performance(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time Elapsed: {time_elapsed:.2f} seconds")
    return result

# Train the Model
print("\nTraining the model...")
print_memory_usage()
measure_performance(train)

# Test the Model
print("\nTesting the model...")
print_memory_usage()
measure_performance(test)
