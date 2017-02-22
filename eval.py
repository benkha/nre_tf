from preprocess import load_data
import tensorflow as tf
import numpy as np


class SentenceEncoder():

    def __init__(self):
        self.stddev = 0.02

        self.data = load_data()
        self.d_a = 50
        self.d_b = 1
        self.d_c = 230
        self.d = self.d_a + self.d_b * 2
        self.l = 3

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

    def train(self):
        sentence = tf.placeholder(tf.float32, [1, None, self.d, 1])
        self.encoder(sentence)

    def encoder(self, x_in):
        with tf.variable_scope('encoder'):
            p_1 = self.conv_2d(x_in, self.l, self.d, self.d_c, "p_1")
            max_pool = tf.nn.max_pool(p_1, ksize=[1, None, self.d_c, 1],
                                      strides=[1, 1, 1, 1],
                                      padding="SAME")
            max_pool_flat = tf.reshape(max_pool, [self.d_c])
            return tf.nn.tanh(max_pool_flat)

    def conv_2d(self, x_in, filter_height, filter_width, out_channels, scope):
        with tf.variable.scope(scope):
            weights = tf.get_variable("weights",
                                      [filter_height, filter_width, 1, out_channels],
                                      initializer=
                                      tf.truncated_normal_initializer(stddev=self.stddev))
            biases = tf.get_variable("biases", [out_channels],
                                     initializer=tf.constant_initializer(0.0))
            return tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding="SAME") + biases







model = SentenceEncoder()
