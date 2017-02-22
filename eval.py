from preprocess import load_data
import tensorflow as tf
import numpy as np


class SentenceEncoder():

    def __init__(self):
        self.data = load_data()
        self.word_dim = 50
        self.word_map = self.data["word_map"]
        self.word_matrix = self.data["word_matrix"]
        self.sentences = self.make_vectors(self.data["train_list"], self.data["train_position_e1"], self.data["train_position_e2"])


    def make_vectors(self, train_list, train_position_e1, train_position_e2):
        sentences = []
        for i in range(len(train_list)):
            word_list = []
            words = train_list[i]
            conl = train_position_e1[i]
            conr = train_position_e2[i]
            for j in range(len(words)):
                word_id = words[j]
                position_e1 = conl[j]
                position_e2 = conr[j]
                word_embed = self.word_matrix[word_id]
                new_embed = np.append(word_embed, [position_e1, position_e2])
                word_list.append(new_embed)
            sentences.append(word_list)
        return sentences






model = SentenceEncoder()
