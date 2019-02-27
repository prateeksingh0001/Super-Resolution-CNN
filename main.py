from dataset import dataset
import numpy as np
import tensorflow as tf
from srcnn import srcnn
import pickle


pickle_in = open("data.pickle", "rb")
data = pickle.load(pickle_in)

srcnn_model = srcnn()

srcnn_model.train(data, 40, 30, 2000)
