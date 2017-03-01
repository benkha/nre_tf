from preprocess import load_data
import tensorflow as tf
import numpy as np


class NeuralRelationExtractor():

    def __init__(self):
        self.stddev = 0.02

        self.data = load_data()
        self.d_a = 50
        self.d_b = 5
        self.d_c = 230
        self.d = self.d_a + self.d_b * 2
        self.l = 3
        self.n_r = len(self.data["relation_list"])

        self.word_map = self.data["word_map"]
        self.word_matrix = self.data["word_matrix"]
        self.sentences = self.make_vectors(self.data["train_list"], self.data["train_position_e1"],
                                           self.data["train_position_e2"])
        self.bags_train = self.data["bags_train"]
        self.find_longest_bag(self.bags_train)
        self.max_length = self.data["max_length"]
        self.num_positions = 2 * self.data["limit"] + 1

        self.sentences = tf.placeholder(tf.float32, [None, self.max_length, 3, 1])
        self.sentence_vectors = self.train_sentence(self.sentences)

        self.logits = self.avg_bag(self.sentence_vectors)

        self.cost =tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.labels)

        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)

    def find_longest_bag(self, bags_train):
        count = 0
        print(len(bags_train))
        for bag in bags_train:
            count = max(count, len(bags_train[bag]))

        print(count)


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
                word_embed = [word_id]
                new_embed = np.append(word_embed, [position_e1, position_e2])
                word_list.append(new_embed)
            sentences.append(word_list)
        return sentences

    def train_sentence(self, sentences):
        word_embedding = tf.constant(self.word_matrix, dtype="float32")
        pad_embedding = tf.get_variable("pad_embedding", [1, self.d_a], dtype="float32", initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        combined_embedding = tf.concat([word_embedding, pad_embedding], 0)

        sentences = tf.to_int64(sentences)
        sentence_embedding = tf.nn.embedding_lookup(combined_embedding, tf.slice(sentences, [0, 0, 0, 0], [-1, -1, 1, -1]))

        position_embedding = tf.get_variable("position_embedding", [self.num_positions, self.d_b])
        position_1 = tf.nn.embedding_lookup(position_embedding, tf.slice(sentences, [0,0, 1, 0], [-1, -1, 1, -1]))
        position_2 = tf.nn.embedding_lookup(position_embedding, tf.slice(sentences, [0,0, 2, 0], [-1, -1, 1, -1]))

        sentence_vector = tf.concat([sentence_embedding, position_1, position_2], 4)
        sentence_vector = tf.reshape(sentence_vector, [-1, 134, self.d, 1])
        sentence_vector = self.encoder(sentence_vector)

        return sentence_vector

    def avg_bag(self, x_in):
        x_flat = tf.reshape(x_in, [-1, self.d_c])

        s = tf.reduce_mean(x_flat, 0)
        s = tf.reshape([1, self.d_c])
        return s

    def fully_connected(self, x_in, input_shape, output_shape, scope):
        with tf.variable_scope(scope):
            matrix = tf.get_variable("matrix", [input_shape, output_shape],
                                     tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias = tf.get_variable("bias", [1, output_shape],
                                   initializer=tf.constant_initializer(0.0))
            return tf.matmul(x_in, matrix) + bias

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                batch = self.get_batch(i)
                entropy_loss = sess.run((self.optimizer, self.cost))

    def encoder(self, x_in):
        with tf.variable_scope('encoder'):
            p_1 = self.conv_2d(x_in, self.l, self.d, self.d_c, "p_1")
            max_pool = tf.nn.max_pool(p_1, ksize=[1, p_1.shape[1], 1, 1],
                                      strides=[1, 1, 1, 1],
                                      padding="VALID")
            return tf.nn.tanh(max_pool)

    def conv_2d(self, x_in, filter_height, filter_width, out_shape, scope):
        with tf.variable_scope(scope):
            weights = tf.get_variable("weights",
                                      [filter_height, filter_width, 1, out_shape],
                                      initializer=
                                      tf.truncated_normal_initializer(stddev=self.stddev))
            biases = tf.get_variable("biases", [out_shape],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding="VALID")
            return tf.nn.bias_add(conv, biases)

    def sentence_attention():
        return None

model = NeuralRelationExtractor()
