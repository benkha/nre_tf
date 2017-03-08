import numpy as np
import struct

bags_train = {}
bags_test = {}
limit = 30
train_list = []
test_list = []
relation_map = {}
fix_len = 70

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

        word_matrix = np.zeros(((word_total + 1), word_dim))
        word_mapping = {}
        word_list = [None] * (word_total + 1)

        for i in range(1, word_total + 1):
            word = read_word(vec_file)

            word_vec = []
            for _ in range(word_dim):
                num = struct.unpack('f', vec_file.read(4))[0]
                word_vec.append(num)
            word_vec = np.array(word_vec)
            norm = np.linalg.norm(word_vec)
            word_matrix[i] = word_vec / norm

            word_mapping[word] = i
            word_list[i] = word

        return word_matrix, word_mapping, word_list

def read_relation():

    with open('data/RE/relation2id.txt', 'r') as f:
        for line in f:
            relation_line = line.split()
            relation = relation_line[0]
            relation_id = int(relation_line[1])
            relation_map[relation] = relation_id
            relation_list.append(relation)
    print("Relation total: ", len(relation_list))

def read_train(word_map, relation_map):
    with open('data/RE/train.txt') as f:
        count = 0
        for line in f:
            # if count > 100000:
            #     break
            count += 1
            words = line.split()

            head_s = words[2]
            tail_s = words[3]
            relation = words[4]

            bags_train.setdefault(head_s + '\t' + tail_s + '\t' + relation, []).append(len(train_list))
            n = 0
            left_num = 0
            rightnum = 0

            sentence = words[5:-1]

            for i in range(len(sentence)):
                if sentence[i] == head_s:
                    left_num = i
                if sentence[i] == tails_s:
                    right_num = i

            output = []

            for i in range(fixlen):
                word = word_map['BLANK']
                rel_e1 = set_with_limit(left_num - i, limit)
                rel_e2 = set_with_limit(right_num - i, limit)
                output.append([word,rel_e1,rel_e2])

            for i in range(min(fixlen,len(sentence))):
                if sentence[i] not in word_map:
                        word = word_map['UNK']
                else:
                        word = word_map[sentence[i]]

                output[i][0] = word

            train_list.append(output)

def read_test(word_map, relation_map):
    with open('data/RE/test.txt') as f:
        count = 0
        for line in f:
            # if count > 100000:
            #     break
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
                if sentence[i] == tails_s:
                    right_num = i

            output = []

            for i in range(fixlen):
                word = word_map['BLANK']
                rel_e1 = set_with_limit(left_num - i, limit)
                rel_e2 = set_with_limit(right_num - i, limit)
                output.append([word,rel_e1,rel_e2])

            for i in range(min(fixlen,len(sentence))):
                if sentence[i] not in word_map:
                        word = word_map['UNK']
                else:
                        word = word_map[sentence[i]]

                output[i][0] = word

            train_test.append(output)

def set_with_limit(value, limit, append=False):
    if value >= limit:
        value = limit
    elif value <= -limit:
        value = -limit
    if append:
        lst.append(value)
    else:
        return value

def make_vectors(sentence_indices, data):
    sentences = []
    for i in sentence_indices:
        sentences.append(data[i])
    return sentences


def next_batch(batch_size, bags_list, data):
    last = 0
    while True:
        next = (last + batch_size)
        wrap = False
        if next > len(bags_list):
            wrap = True
        sentence_indices = []
        bag_labels = []
        bag_indices = []
        for i in range(last, min(next, len(bags_list))):
            bag_name = bags_list[i]
            bag_labels.append(relation_map.get(bag_name.split()[2], 0))
            sentence_indices.append(bags_train[bag_name])
            bag_indices.append(len(bags_train[bag_name]))
        if wrap:
            for i in range(next % len(bags_list)):
                bag_name = bags_list[i]
                bag_labels.append(relation_map.get(bag_name.split()[2], 0))
                sentence_indices.append(bags_train[bag_name])
                bag_indices.append(len(bags_train[bag_name]))

        flat_indices = [index for sublist in sentence_indices for index in sublist]
        yield make_vectors(flat_indices, data), bag_labels, bag_indices
        last = (next % len(bags_list))

def load_data():
    word_matrix, word_map, word_list = read_vec()
    read_relation()
    read_train(word_map, relation_map)
    bags_list = list(bags_train.keys())
    max_length = fixlen
    print("Max Length", max_length)
    data = {
        "word_matrix" : word_matrix,
        "word_map": word_map,
        "relation_map": relation_map,
        "bags_train": bags_train,
        "bags_test", bags_test
        "bags_list": bags_list,
        "train_list":, train_list,
        "test_list":, test_list
        "max_length": max_length,
        "limit": limit
    }
    return data
