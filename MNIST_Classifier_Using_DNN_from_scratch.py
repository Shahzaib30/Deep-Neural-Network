import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ReLU(x):
    return np.maximum(0,x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def load_mnist_data():
    