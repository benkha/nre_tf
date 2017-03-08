from preprocess import load_data, next_batch
import tensorflow as tf
import numpy as np


class NeuralRelationExtractor():

    def __init__(self):
        self.stddev = 0.02
        self.batch_size = 160

        self.data = load_data()
        self.d_a = 50
        self.d_b = 5
        self.d_c = 230
        self.d = self.d_a + self.d_b * 2
        self.l = 3
        self.n_r = len(self.data["relation_list"])

        self.word_map = self.data["word_map"]
        self.word_matrix = self.data["word_matrix"]
        self.num_positions = 2 * self.data["limit"] + 1
        self.bags_list = self.data["bags_list"]
        self.max_length = self.data["max_length"]

        self.sentences_placeholder = tf.placeholder(tf.float32, [None, self.max_length, 3])
        self.sentences = tf.expand_dims(self.sentences_placeholder,  -1)
        self.sentence_vectors = self.train_sentence(self.sentences)

        self.flat_sentences = tf.squeeze(self.sentence_vectors, [1, 2])
        self.bag_indices = tf.placeholder(tf.int32, [self.batch_size])
        self.logits = self.avg_bags(self.bag_indices, self.flat_sentences)

        # self.logits = self.avg_bag(self.sentence_vectors)
        self.labels_placeholder = tf.placeholder(tf.int32, [None])

        self.cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.logits))

        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)

    def avg_bags(self, bag_indices, sentence_vectors):
        prev = 0
        means = []
        for j in range(160):
            i = bag_indices[j]
            sum_tensors = tf.slice(sentence_vectors, [prev, 0], [i, -1])
            mean_tensors = tf.reduce_mean(sum_tensors, 0)
            means.append(mean_tensors)
            prev += i
        return tf.stack(means)


    def find_longest_bag(self, bags_train):
        count = 0
        print(len(bags_train))
        for bag in bags_train:
            count = max(count, len(bags_train[bag]))

        print(count)

    def train_sentence(self, sentences):
        word_embedding = tf.constant(self.word_matrix, dtype="float32")
        pad_embedding = tf.get_variable("pad_embedding", [1, self.d_a], dtype="float32", initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        combined_embedding = tf.concat([word_embedding, pad_embedding], 0)

        sentences = tf.to_int64(sentences)
        sentence_embedding = tf.nn.embedding_lookup(combined_embedding, tf.slice(sentences, [0, 0, 0, 0], [-1, -1, 1, -1]))

        position_embedding = tf.get_variable("position_embedding", [self.num_positions, self.d_b], initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        position_1 = tf.nn.embedding_lookup(position_embedding, tf.slice(sentences, [0,0, 1, 0], [-1, -1, 1, -1]))
        position_2 = tf.nn.embedding_lookup(position_embedding, tf.slice(sentences, [0,0, 2, 0], [-1, -1, 1, -1]))

        sentence_vector = tf.concat([sentence_embedding, position_1, position_2], 4)
        sentence_vector = tf.squeeze(sentence_vector, [2, 3])
        sentence_vector = tf.expand_dims(sentence_vector, -1)
        sentence_vector = self.encoder(sentence_vector)

        return sentence_vector

    def avg_bag(self, x_in):
        x_flat = tf.reshape(x_in, [-1, self.d_c])

        s = tf.reduce_mean(x_flat, 0)
        s = tf.reshape(s, [1, self.d_c])
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
        self.batch_iter = next_batch(self.batch_size, self.bags_list, self.word_matrix, self.max_length)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10):
                sentences, bag_labels, bag_indices = next(self.batch_iter)
                entropy_loss = sess.run((self.optimizer, self.cost), feed_dict={self.sentences_placeholder: sentences, self.labels_placeholder: bag_labels, self.bag_indices: bag_indices})
                print("Epoch:", epoch)
                print("Entropy Loss", entropy_loss)

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
model.train()
