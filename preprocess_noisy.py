import sys
import struct
import random

import numpy as np
import sklearn
import pickle

bags_train = {}
bags_test = {}
limit = 30
train_list = []
train_labels = []
test_list = []
relation_map = {}
fix_len = 70
left_num_train = []
right_num_train = []

def read_word(f):
    word = b''
    not_space = True
    while (not_space):
        c = f.read(1)
        if (c == b' '):
            not_space = False
        elif (c != b'\n'):
            word += c
    return word.decode('utf-8')

def read_vec():
    with open('./data/vec.bin', 'rb') as vec_file:
        dim = vec_file.readline()
        dims = dim.split()
        word_total = int(dims[0])
        word_dim = int(dims[1])
        print('Word total:', word_total)
        print('Word dimension:', word_dim)

        word_matrix = np.zeros(((word_total + 1), word_dim), dtype="float32")
        word_map = {}

        for i in range(1, word_total + 1):
            word = read_word(vec_file)

            word_vec = []
            for _ in range(word_dim):
                num = struct.unpack('f', vec_file.read(4))[0]
                word_vec.append(num)
            word_vec = np.array(word_vec)
            norm = np.linalg.norm(word_vec)
            word_matrix[i] = word_vec / norm

            word_map[word] = i

        word_map['UNK'] = len(word_map)
        word_map['BLANK'] = len(word_map)

        return word_matrix, word_map

def read_relation():

    with open('data/RE/relation2id.txt', 'r') as f:
        for line in f:
            relation_line = line.split()
            relation = relation_line[0]
            relation_id = int(relation_line[1])
            relation_map[relation] = relation_id
        relation_map['UNK'] = len(relation_map)
    print("Relation total: ", len(relation_map))

def read_train(word_map, relation_map):
    with open('data/RE/train.txt') as f:
        count = 0
        for line in f:
            # if count > 1000:
            #     break
            # if count % 10000 == 0:
            #     print("Count:", count)
            #     print(sys.getsizeof(train_list) / 1e6)

            count += 1
            words = line.split()

            head_s = words[2]
            tail_s = words[3]
            relation = words[4]

            # bags_train.setdefault(head_s + '\t' + tail_s + '\t' + relation, []).append(len(train_list))
            if relation in relation_map:
                train_labels.append(relation_map[relation])
            else:
                train_labels.append(relation_map['UNK'])
            n = 0
            left_num = 0
            rightnum = 0

            sentence = words[5:-1]

            for i in range(len(sentence)):
                if sentence[i] == head_s:
                    left_num = i
                if sentence[i] == tail_s:
                    right_num = i

            left_num_train.append(left_num)
            right_num_train.append(right_num)

            min_len = min(fix_len,len(sentence))
            output = np.zeros((min_len, 3), dtype="int32")

            for i in range(min(fix_len,len(sentence))):
                rel_e1 = set_with_limit(left_num - i, limit) + limit
                rel_e2 = set_with_limit(right_num - i, limit) + limit
                if sentence[i] not in word_map:
                        word = word_map['UNK']
                else:
                        word = word_map[sentence[i]]
                output[i] = np.array([word,rel_e1,rel_e2])
            train_list.append(output)
    print("Dumping picke data")
    pickle.dump(train_list, open("train_list.pickle", "wb"))
    pickle.dump(train_labels, open("train_labels.pickle", "wb"))
    pickle.dump(left_num_train, open("left_num_train.pickle", "wb"))
    pickle.dump(right_num_train, open("right_num_train.pickle", "wb"))

def read_test(word_map, relation_map):
    with open('data/RE/test.txt') as f:
        count = 0
        for line in f:
            # if count > 100000:
            #     break
            if count % 10000 == 0:
                print("Count:", count)
            count += 1
            words = line.split()

            head_s = words[2]
            tail_s = words[3]
            relation = words[4]

            bags_test.setdefault(head_s + '\t' + tail_s + '\t' + relation, []).append(len(test_list))
            n = 0
            left_num = 0
            rightnum = 0

            sentence = words[5:-1]

            for i in range(len(sentence)):
                if sentence[i] == head_s:
                    left_num = i
                if sentence[i] == tail_s:
                    right_num = i

            min_len = min(fix_len,len(sentence))
            output = np.zeros((min_len, 3), dtype="int32")

            # for i in range(fix_len):
            #     word = word_map['BLANK']
            #     rel_e1 = set_with_limit(left_num - i, limit)
            #     rel_e2 = set_with_limit(right_num - i, limit)
            #     output.append([word,rel_e1,rel_e2])

            for i in range(min(fix_len,len(sentence))):
                rel_e1 = set_with_limit(left_num - i, limit) + limit
                rel_e2 = set_with_limit(right_num - i, limit) + limit
                if sentence[i] not in word_map:
                        word = word_map['UNK']
                else:
                        word = word_map[sentence[i]]
                output[i] = np.array([word,rel_e1,rel_e2])

            test_list.append(output)



def set_with_limit(value, limit):
    if value >= limit:
        value = limit
    elif value <= -limit:
        value = -limit
    return value

def make_vectors(sentence_indices, data, word_map):
    sentences = []
    for i in sentence_indices:
        sentence = data[i]
        if len(sentence) < fix_len:
            new_sentence = np.zeros((fix_len, 3), dtype="int32")
            new_sentence[:len(sentence), :] = sentence
            for j in range(len(sentence), fix_len):
                word = word_map['BLANK']
                left_num = left_num_train[i]
                right_num = right_num_train[i]
                rel_e1 = set_with_limit(left_num - j, limit) + limit
                rel_e2 = set_with_limit(right_num - j, limit) + limit
                new_sentence[j] = [word,rel_e1,rel_e2]
            sentence = new_sentence

        sentences.append(sentence)
    return sentences


def next_batch(batch_size, data, labels, word_map):
    last = 0
    data, labels = sklearn.utils.shuffle(data, labels)
    while True:
        next = (last + batch_size)
        wrap = False
        if next > len(data):
            wrap = True
        sentence_indices = []
        sentence_labels = []
        for i in range(last, min(next, len(data))):
            sentence_labels.append(labels[i])
            sentence_indices.append(i)
        if wrap:
            for i in range(next % len(train_list)):
                sentence_labels.append(labels[i])
                sentence_indices.append(i)
        vectors = make_vectors(sentence_indices, data, word_map)
        # print('Size of batch (MB):', sys.getsizeof(vectors) / 1e6)
        yield vectors, sentence_labels
        last = (next % len(train_list))

def compute_average_bag(bags):
    total = 0.0
    max_len = 0.0
    for bag in bags:
        total += len(bags[bag])
        max_len = max(max_len, len(bags[bag]))
    return total / len(bags), max_len


def load_data():
    global train_list, train_labels, left_num_train, right_num_train
    word_matrix, word_map = read_vec()
    read_relation()
    print("=====Starting to read training data=====")
    try:
        train_list = pickle.load(open("train_list.pickle", "rb"))
        train_labels = pickle.load(open("train_labels.pickle", "rb"))
        left_num_train = pickle.load(open("left_num_train.pickle", "rb"))
        right_num_train = pickle.load(open("right_num_train.pickle", "rb"))
    except (OSError, IOError) as e:
        print("Error loading pickle data")
        read_train(word_map, relation_map)
    print("Size of train_list (MB):", sys.getsizeof(train_list) / 1e6)
    # bags_list = list(bags_train.keys())
    # random.shuffle(bags_list)
    # avg_len, max_len = compute_average_bag(bags_train)
    # print("Length of average bag:", avg_len)
    # print("Length of max bag:", max_len)

    max_length = fix_len
    data = {
        "word_matrix" : word_matrix,
        "word_map": word_map,
        "relation_map": relation_map,
        "train_list": train_list,
        "train_labels": train_labels,
        # "test_list": test_list,
        "max_length": max_length,
        "limit": limit
    }
    return data

if __name__ == "__main__":
    load_data()
