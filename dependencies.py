# importing the dependencies
import keras
import numpy as np
import itertools
import tensorflow as tf
from keras import backend as k
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
from IPython.display import clear_output
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense,Flatten
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential ,Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense,Dropout
from keras.callbacks import ModelCheckpoint
