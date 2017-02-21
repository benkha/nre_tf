import numpy as np
import binascii
import struct

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
            id = int(relation_line[1])
            relation_map[relation] = id
            relation_list.append(relation)
    print("Relation total: ", len(relation_list))
    return relation_map, relation_list

# word_matrix, word_mapping, word_list = read_vec()
# relation_map, relation_list = read_relation()
