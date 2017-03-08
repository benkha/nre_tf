import numpy as np
import struct

bags_train = {}
head_list = []
tail_list = []
relation_num_list = []
train_length = []
limit = 30
train_list = []
train_position_e1 = []
train_position_e2 = []
left_num_list = []
right_num_list = []
relation_map = {}
relation_list = []

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
    position_min_e1 = position_max_e1 = position_min_e2 = position_max_e2 = 0

    with open('data/RE/train.txt') as f:
        count = 0
        for line in f:
            # if count > 10000:
            #     break
            count += 1
            words = line.split()
            head_s = words[2]
            head = word_map.get(head_s, 0)
            tail_s = words[3]
            tail = word_map.get(tail_s, 0)
            relation = words[4]
            bags_train.setdefault(head_s + '\t' + tail_s + '\t' + relation, []).append(len(head_list))
            relation_id = relation_map.get(relation, 0)
            tmpp = []
            n = 0
            left_num = 0
            rightnum = 0
            for i in range(5, len(words)):
                temp_word = words[i]
                if (temp_word == '###END###'):
                    break
                word_id = word_map.get(temp_word, 0)
                if (temp_word == head_s):
                    left_num = n
                if (temp_word == tail_s):
                    right_num = n
                n += 1
                tmpp.append(word_id)
            left_num_list.append(left_num)
            right_num_list.append(right_num)
            train_length.append(n)
            con = np.zeros(n, dtype="int32")
            conl = np.zeros(n, dtype="int8")
            conr = np.zeros(n, dtype="int8")
            for i in range(n):
                con[i] = tmpp[i]
                set_with_limit(conl, i, left_num - i, limit)
                set_with_limit(conr, i, right_num - i, limit)
            train_list.append(con)
            train_position_e1.append(conl)
            train_position_e2.append(conr)

    position_total_e1 = position_max_e1 - position_min_e1 + 1
    position_total_e2 = position_max_e2 - position_min_e2 + 1
    return position_total_e1, position_total_e2, left_num_list, right_num_list

def set_with_limit(lst, i, value, limit, append=False):
    if value >= limit:
        value = limit
    elif value <= -limit:
        value = -limit
    if append:
        lst.append(value)
    else:
        lst[i] = value

def make_vectors(max_length, sentence_indices, word_matrix):
    sentences = []
    for i in sentence_indices:
        word_list = []
        words = list(train_list[i])
        conl = list(train_position_e1[i])
        conr = list(train_position_e2[i])
        left_num = left_num_list[i]
        right_num = right_num_list[i]
        if len(words) < max_length:
            for j in range(len(words), max_length):
                words.append(len(word_matrix))
                set_with_limit(conl, j, left_num - j, limit, append=True)
                set_with_limit(conr, j, right_num - j, limit, append=True)
        for j in range(len(words)):
            word_id = words[j]
            position_e1 = conl[j] + limit
            position_e2 = conr[j] + limit
            word_embed = [word_id]
            new_embed = np.append(word_embed, [position_e1, position_e2])
            word_list.append(new_embed)
        sentences.append(word_list)
    return sentences


def next_batch(batch_size, bags_list, word_matrix, max_length):
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
        yield make_vectors(max_length, flat_indices, word_matrix), bag_labels, bag_indices
        last = (next % len(bags_list))

def load_data():
    word_matrix, word_map, word_list = read_vec()
    read_relation()
    read_train(word_map, relation_map)
    bags_list = list(bags_train.keys())
    max_length = max(train_length)
    print("Max Length", max_length)
    data = {
        "word_matrix" : word_matrix,
        "word_map": word_map,
        "word_list": word_list,
        "relation_map": relation_map,
        "relation_list": relation_list,
        "bags_train": bags_train,
        "bags_list": bags_list,
        "max_length": max_length,
        "limit": limit
    }
    return data
