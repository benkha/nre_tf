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
    relation_map = {}
    relation_list = []
    with open('data/RE/relation2id.txt', 'r') as f:
        for line in f:
            relation_line = line.split()
            relation = relation_line[0]
            relation_id = int(relation_line[1])
            relation_map[relation] = relation_id
            relation_list.append(relation)
    print("Relation total: ", len(relation_list))
    return relation_map, relation_list

def read_train(word_map, relation_map):
    position_min_e1 = position_max_e1 = position_min_e2 = position_max_e2 = 0
    with open('data/RE/train.txt') as f:
        for line in f:
            words = line.split()
            head_s = words[2]
            head = word_map.get(head_s, 0)
            tail_s = words[3]
            tail = word_map.get(tail_s, 0)
            relation = words[4]
            bags_train.setdefault(head_s + '\t' + tail_s + 's' + relation, []).append(len(head_list))
            num = relation_map.get(relation, 0)
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
            head_list.append(head)
            tail_list.append(tail)
            relation_num_list.append(num)
            train_length.append(n)
            con = [0] * n
            conl = [0] * n
            conr = [0] * n
            for i in range(n):
                con[i] = tmpp[i]
                set_with_limit(conl, i, left_num - i, limit)
                set_with_limit(conr, i, right_num - i, limit)
                position_max_e1 = max(position_max_e1, conl[i])
                position_max_e2 = max(position_max_e2, conr[i])
                position_min_e1 = min(position_min_e1, conl[i])
                position_min_e2 = min(position_min_e2, conr[i])
            train_list.append(con)
            train_position_e1.append(conl)
            train_position_e2.append(conr)

    for i in range(len(train_position_e1)):
        train_len = train_length[i]
        work_1 = train_position_e1[i]
        work_2 = train_position_e2[i]

        for j in range(train_len):
            work_1[j] -= position_min_e1
            work_2[j] -= position_min_e2
    position_total_e1 = position_max_e1 - position_min_e1 + 1
    position_total_e2 = position_max_e2 - position_min_e2 + 1
    return position_total_e1, position_total_e2

def set_with_limit(lst, i, value, limit):
    if value >= limit:
        value = limit
    elif value <= -limit:
        value = -limit
    lst[i] = value


def load_data():
    word_matrix, word_map, word_list = read_vec()
    relation_map, relation_list = read_relation()
    position_total_e1, position_total_e2 = read_train(word_map, relation_map)
