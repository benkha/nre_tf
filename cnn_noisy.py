from preprocess_noisy import load_data, next_batch
import tensorflow as tf
import os
import time
import math
import numpy as np
import datetime
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import pickle
from itertools import cycle

def _summary_for_scalar(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=float(value))])

class NeuralRelationExtractor():


    def __init__(self):
        self.stddev = 0.02
        self.batch_size = 128
        self.test_length = 2048

        self.data = load_data()
        self.d_a = 50
        self.d_b = 5
        self.d_c = 230
        self.d = self.d_a + self.d_b * 2
        self.l = 3
        self.n_r = len(self.data["relation_map"])

        self.word_map = self.data["word_map"]
        self.word_matrix = self.data["word_matrix"]

        self.train_list = self.data["train_list"]
        self.train_labels = self.data["train_labels"]
        self.left_num_train = self.data["left_num_train"]
        self.right_num_train = self.data["right_num_train"]

        self.test_list = self.data["test_list"]
        self.test_labels = self.data["test_labels"]
        self.left_num_test = self.data["left_num_test"]
        self.right_num_test = self.data["right_num_test"]

        self.num_positions = 2 * self.data["limit"] + 1
        self.num_epochs = 1
        self.max_length = self.data["max_length"]

        self.sentences_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.max_length, 3])
        self.sentences = tf.expand_dims(self.sentences_placeholder,  -1)
        self.sentence_vectors = self.train_sentence(self.sentences)

        self.flat_sentences = tf.squeeze(self.sentence_vectors, [1, 2])
        self.logits = self.fully_connected(self.flat_sentences, self.d_c, self.n_r, "logits")
        self.probabilities = tf.nn.softmax(self.logits)
        self.labels_placeholder = tf.placeholder(tf.int32, [self.batch_size])

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=self.logits))

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost, global_step=self.global_step)

        tf.summary.scalar("loss", self.cost)

        self.train_accuracy = self.accuracy(self.logits, self.labels_placeholder)
        self.summary_op = tf.summary.merge_all()

    def accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.to_int32(tf.argmax(logits, 1)), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("auc", accuracy)
        return accuracy

    def fully_connected(self, x_in, input_shape, output_shape, scope):
        with tf.variable_scope(scope):
            matrix = tf.get_variable("matrix", [input_shape, output_shape],
                                     tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias = tf.get_variable("bias", [1, output_shape],
                                   tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            return tf.matmul(x_in, matrix) + bias

    def avg_bags(self, bag_indices, sentence_vectors):
        prev = 0
        means = []
        for j in range(self.batch_size):
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

        sentence_embedding = tf.nn.embedding_lookup(combined_embedding, tf.slice(sentences, [0, 0, 0, 0], [-1, -1, 1, -1]))

        position_embedding = tf.get_variable("position_embedding", [self.num_positions, self.d_b], dtype="float32", initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        position_1 = tf.nn.embedding_lookup(position_embedding, tf.slice(sentences, [0,0, 1, 0], [-1, -1, 1, -1]))
        position_2 = tf.nn.embedding_lookup(position_embedding, tf.slice(sentences, [0,0, 2, 0], [-1, -1, 1, -1]))

        sentence_vector = tf.concat([sentence_embedding, position_1, position_2], 4)
        sentence_vector = tf.squeeze(sentence_vector, [2, 3])
        sentence_vector = tf.expand_dims(sentence_vector, -1)
        sentence_vector = self.encoder(sentence_vector)

        return sentence_vector



    def test_step(self, sess, length=None):
        dev_loss = []
        dev_auc = []
        dev_probabilities = []
        dev_labels = []
        if length is None:
            length = self.test_length

        # create batch
        test_iter = next_batch(self.batch_size, self.test_list, self.test_labels, self.left_num_test, self.right_num_test, self.word_map, length, test=True)

        for batch in test_iter:
            sentences, labels = batch
            #a_batch = np.ones((len(batch), 1), dtype=np.float32) / len(batch) # average
            loss, accuracy, _, probability = sess.run([self.cost, self.train_accuracy, self.summary_op, self.probabilities], feed_dict={self.sentences_placeholder: sentences, self.labels_placeholder: labels})
            dev_loss.append(loss)
            dev_auc.append(accuracy)
            dev_probabilities.append(probability)
            dev_labels.append(labels)

        return np.mean(dev_loss), np.mean(dev_auc), dev_probabilities, dev_labels

    def train(self):
        self.batch_iter = next_batch(self.batch_size, self.train_list, self.train_labels, self.left_num_train, self.right_num_train, self.word_map, len(self.train_list))
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        timestamp = str(int(time.time()))
        save_path = './sample_model/' + timestamp + '/'
        print("Timestamp", timestamp)
        tensor_board_dir = './tensorboard/' + timestamp + '/train'
        tensor_board_test_dir = './tensorboard/' + timestamp + '/test'
        if not os.path.exists(tensor_board_dir):
            os.makedirs(tensor_board_dir)
        if not os.path.exists(tensor_board_test_dir):
            os.makedirs(tensor_board_test_dir)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(tensor_board_dir, sess.graph)
            test_writer = tf.summary.FileWriter(tensor_board_test_dir, sess.graph)
            saver = tf.train.Saver(max_to_keep=None)
            # saver.restore(sess, save_path + 'CNN_NOISY_model-7001')
            print("Total iterations:", self.num_epochs * len(self.train_list) // self.batch_size)
            # for step in range(self.num_epochs * len(self.train_list) // self.batch_size):
            for step in range(1000):
                sentences, sentence_labels = next(self.batch_iter)
                if step == 0:
                    dev_loss, dev_auc, _, _ = self.test_step(sess)
                _, loss, accuracy, summary = sess.run((self.optimizer, self.cost, self.train_accuracy, self.summary_op),
                                                      feed_dict={self.sentences_placeholder: sentences, self.labels_placeholder: sentence_labels})
                if step % 50 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("Train: step {}, loss {:g}, accuracy {:g}".format(step, loss, accuracy))
                    train_writer.add_summary(summary, global_step=step)
                    if step != 0:
                        dev_loss, dev_auc, _, _ = self.test_step(sess)
                    print("Test: step {}, loss {:g}, accuracy {:g}".format(step, dev_loss, dev_auc))
                    test_writer.add_summary(_summary_for_scalar('loss', dev_loss), global_step=step)
                    test_writer.add_summary(_summary_for_scalar('auc', dev_auc), global_step=step)
                if step % 1000 == 0:
                    path = saver.save(sess,save_path +'CNN_NOISY_model',global_step=self.global_step)
            self.test(sess)


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
                                      dtype="float32",
                                      initializer=
                                      tf.truncated_normal_initializer(stddev=self.stddev))
            biases = tf.get_variable("biases", [out_shape],
                                     dtype="float32",
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding="VALID")
            return tf.nn.bias_add(conv, biases)

    def attention_bags(self, bag_indices, sentence_vectors):
        prev = 0
        means = []
        for j in range(160):
            i = bag_indices[j]
            bag_tensors = tf.slice(sentence_vectors, [prev, 0], [i, -1])
            mean_tensors = tf.reduce_mean(sum_tensors, 0)
            means.append(mean_tensors)
            prev += i
        return tf.stack(means)

    def eval_step(self, sess, length=None):
        dev_loss = []
        dev_auc = []
        dev_probabilities = []
        dev_labels = []
        if length is None:
            length = self.test_length

        # create batch
        test_iter = eval_batch()

        for batch in test_iter:
            sentences, labels = batch
            #a_batch = np.ones((len(batch), 1), dtype=np.float32) / len(batch) # average
            loss, accuracy, _, probability = sess.run([self.cost, self.train_accuracy, self.summary_op, self.probabilities], feed_dict={self.sentences_placeholder: sentences, self.labels_placeholder: labels})
            dev_loss.append(loss)
            dev_auc.append(accuracy)
            dev_probabilities.append(np.max(probability, axis=0))
            dev_labels.append(labels[0])

        return np.mean(dev_loss), np.mean(dev_auc), dev_probabilities, dev_labels

    def test(self, sess=None):
        print("===Starting testing===")
        if sess != None:
            print("Test length:", len(self.test_list))
            # loss, auc, probabilities, labels = self.test_step(sess, (len(self.test_list) // self.batch_size) * self.batch_size)
            loss, auc, probabilities, labels = self.eval_step(sess)
            probabilities = np.concatenate(probabilities, axis=0)
            labels = np.concatenate(labels, axis=0)
            print("Dumping pr curve")
            pickle.dump(probabilities, open("./pickle/pr_curve/noisy_p.pickle", "wb"))
            pickle.dump(labels, open("./pickle/pr_curve/noisy_label.pickle", "wb"))
            self.generate_pr(labels, probabilities)
        else:
            print("Loading pr curve")
            probabilities = pickle.load(open("./pickle2/pr_curve/noisy_p.pickle", "rb"))
            labels = pickle.load(open("./pickle2/pr_curve/noisy_label.pickle", "rb"))
            print("Test length:", len(labels))
            self.generate_pr(labels, probabilities)

    def generate_y_matrix(self, y_test):
        y_matrix = np.zeros((len(y_test), self.n_r))
        for i in range(len(y_test)):
            y_matrix[i][y_test[i]] = 1
        return y_matrix

    def generate_pr(self, y_test, y_score):
        precision = dict()
        recall = dict()
        average_precision = dict()
        plt.hist(y_test)
        plt.show()
        print(y_test)
        y_test = self.generate_y_matrix(y_test)
        y_score_max = np.argmax(y_score, axis=1)
        print(y_score_max)
        plt.hist(y_score_max)
        plt.show()
        y_score = y_score[:, 1:]
        y_test = y_test[:, 1:]
        # for i in range(self.n_r - 1):
        #     precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
        #                                                 y_score[:, i])
        #     average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])


        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                        y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score,
                                                             average="micro")
        lw = 2
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.clf()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
        # for i, color in zip(range(self.n_r - 1), colors):
        #     if not math.isnan(average_precision[i]):
        #         plt.plot(recall[i], precision[i], color=color, lw=lw,
        #                  label='Precision-recall curve of class {0} (area = {1:0.2f})'
        #                  ''.format(i, average_precision[i]))
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()


    def sentence_attention(self, x_in):
        A = tf.get_variable("A", [self.d_c],
                            initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        r = tf.get_variable("r", [self.d_c, self.n_r],
                            initializer=tf.truncated_normal_initializer(stddev=self.stddev))
        A_matrix = tf.diag(A)
        return tf.matmul(x_in, tf.matmul(A_matrix, r))


model = NeuralRelationExtractor()
print("=====Starting to train=====")
# model.train()
model.test()
