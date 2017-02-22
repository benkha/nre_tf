from preprocess import load_data
import tensorflow as tf
import numpy as np


class SentenceEncoder():

    def __init__(self):
        self.data = load_data()
