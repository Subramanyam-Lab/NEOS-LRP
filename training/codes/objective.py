import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import DeepSetArchitecture
import matplotlib.pyplot as plt
import random
import numpy as np
from train_test_val import master

def run(config):

    output= master(config, metrics=False, exportonnx=False, testing=False, seed=42, N_dim=3)

    return -output