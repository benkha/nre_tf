import numpy as np
import struct
import random
import sys

bags_train = {}
bags_test = {}
limit = 30
train_list = []
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

            bags_train.setdefault(head_s + '\t' + tail_s + '\t' + relation, []).append(len(train_list))
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
                rel_e1 = set_with_limit(left_num - i, limit)
                rel_e2 = set_with_limit(right_num - i, limit)
                if sentence[i] not in word_map:
                        word = word_map['UNK']
                else:
                        word = word_map[sentence[i]]
                output[i] = np.array([word,rel_e1,rel_e2])
            train_list.append(output)

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
                rel_e1 = set_with_limit(left_num - i, limit)
                rel_e2 = set_with_limit(right_num - i, limit)
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
                rel_e1 = set_with_limit(left_num - j, limit)
                rel_e2 = set_with_limit(right_num - j, limit)
                new_sentence[j] = [word,rel_e1,rel_e2]
            sentence = new_sentence

        sentences.append(sentence)
    return sentences


def next_batch(batch_size, bags_list, data, word_map):
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
        vectors = make_vectors(flat_indices, data, word_map)
        # print('Size of batch (MB):', sys.getsizeof(vectors) / 1e6)
        yield vectors, bag_labels, bag_indices
        last = (next % len(bags_list))

def compute_average_bag(bags):
    total = 0.0
    max_len = 0.0
    for bag in bags:
        total += len(bags[bag])
        max_len = max(max_len, len(bags[bag]))
    return total / len(bags), max_len


def load_data():
    word_matrix, word_map = read_vec()
    read_relation()
    print("=====Starting to read training data=====")
    read_train(word_map, relation_map)
    print("Size of train_list (MB):", sys.getsizeof(train_list) / 1e6)
    bags_list = list(bags_train.keys())
    random.shuffle(bags_list)
    avg_len, max_len = compute_average_bag(bags_train)
    print("Length of average bag:", avg_len)
    print("Length of max bag:", max_len)

    max_length = fix_len
    data = {
        "word_matrix" : word_matrix,
        "word_map": word_map,
        "relation_map": relation_map,
        "bags_train": bags_train,
        "bags_test": bags_test,
        "bags_list": bags_list,
        "train_list": train_list,
        "test_list": test_list,
        "max_length": max_length,
        "limit": limit
    }
    return data

if __name__ == "__main__":
    load_data()
