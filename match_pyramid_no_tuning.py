# -*- coding: utf-8 -*-
"""
@author: Parisa Beygi
"""
import json
import operator
import os
import pickle
import resource
import sys
import time

import keras
import numpy
import codecs
import pandas
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, BatchNormalization, Activation
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import dot
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.optimizers import Adagrad, Adam

# import globalvars

from keras import regularizers, callbacks
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# load datasett

config = json.loads(open(sys.argv[1]).read())


def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(t_)
    return numpy.asarray(ids)


def get(file):
    id1s = []
    id2s = []
    labels = []
    for line in file:
        contents = line.strip().split()
        if 'label' not in contents:
            id1s.append(contents[1])
            id2s.append(contents[2])
            labels.append(contents[0])

    return labels, id1s, id2s

def get_sparses(sparse_id_list, file, index):
    id1s = []
    id2s = []
    labels = []
    for line in file:
        contents = line.strip().split()
        if 'label' not in contents:
            if int(contents[index+1]) in sparse_id_list:
                id1s.append(contents[1])
                id2s.append(contents[2])
                labels.append(contents[0])

    return labels, id1s, id2s



def getq(file, size):
    qs = []
    for line in file:
        contents = line.strip().split('\t')
        if 'qid' not in contents:
            # print ('len(contents)')
            # print (len(contents))
            # print (contents[0])
            # if len(contents) > 1:
            qs.append(tuple((contents[0], ' '.join(contents[1].split(' ')[0:size]))))
            # else:
            #     qs.append(tuple((contents[0], '')))

    return qs


def get_qd(question_path, size):
    print ('in get_qd()')
    q_file = codecs.open(question_path, 'r', encoding='utf8')
    qes = getq(q_file, size)
    qes_all_words = {}
    for i in range(len(qes)):
        qes_all_words[qes[i][0]] = qes[i][1]

    return qes_all_words

query_fit_max_len = 400
doc_fit_max_len = 400

data_dir = config['encoded_data_dir']
TRAIN_PATH = os.path.join(data_dir, 'relation_train.txt')
VALID_PATH = os.path.join(data_dir, 'relation_valid.txt')
TEST_PATH = os.path.join(data_dir, 'relation_test.txt')
# QUESTION_PATH = 'data/amazon/patio/music_inst_user_text.txt'
QUESTION_PATH = os.path.join(data_dir, 'user_text.txt')
DOC_PATH = os.path.join(data_dir, 'item_text.txt')

print ('1------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

qes_all_words = get_qd(QUESTION_PATH, query_fit_max_len)
print ('2------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
docs_all_words = get_qd(DOC_PATH, doc_fit_max_len)
print ('3-----ssss-------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

data_processed_dir = config['data_general_processed_dir']
# sparse_dict load
# sparse_user_dict_path = os.path.join(data_processed_dir, 'user_dict.sparse')
# sparse_user_dict_file = open(sparse_user_dict_path, 'rb')
# sparse_user_dict = pickle._Unpickler(sparse_user_dict_file)
# sparse_user_dict.encoding = 'latin1'
# sparse_user_dict = sparse_user_dict.load()
#
# sparse_item_dict_path = os.path.join(data_processed_dir, 'item_dict.sparse')
# sparse_item_dict_file = open(sparse_item_dict_path, 'rb')
# sparse_item_dict = pickle._Unpickler(sparse_item_dict_file)
# sparse_item_dict.encoding = 'latin1'
# sparse_item_dict = sparse_item_dict.load()


# sparse_item_dict_path = os.path.join(general_dir, 'item_dict.sparse')
# sparse_item_dict_file = open(sparse_item_dict_path, 'rb')
# sparse_item_dict = pickle.load(sparse_item_dict_file)

import gc


def get_texts_sparse(file_path, sparse_id_list, index, mode=None):
    #    qes = pd.read_csv(question_path, sep='	', dtype=str)
    #    qes = qes.dropna()
    #    file = pd.read_csv(file_path, sep=' ', dtype=str)
    #    file = file.dropna()
    #    q1id, q2id = file['q1'], file['q2']
    #    id1s, id2s = get_ids(q1id), get_ids(q2id)
    file = codecs.open(file_path, 'r', encoding='utf8')
    # todo: qes_all_words, docs_all_words
    # q_file = codecs.open(question_path, 'r', encoding='utf8')
    # d_file = codecs.open(doc_path, 'r', encoding='utf8')

    labels, id1s, id2s = get_sparses(sparse_id_list, file, index)
    # todo: qes_all_words, docs_all_words
    # qes = getq(q_file)
    # qes_all_words = {}
    # for i in range(len(qes)):
    #     qes_all_words[qes[i][0]] = qes[i][1]
    texts1 = []
    # print (mode)
    # print (len(id1s))
    print ('middle 1 in {}'.format(mode))

    print('4-----ssss-------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    counter = 0
    size_sum = 0
    for t_ in id1s:
        counter +=1
        print (counter)
        print('----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # l = qes_all_words[t_].split(' ')
        l = qes_all_words[t_]
        size_sum += sys.getsizeof(l)
        # if len(l) > query_fit_max_len:
        #     l = l[0:query_fit_max_len]
        texts1.append(l)

    # todo: qes_all_words, docs_all_words
    # docs = getq(d_file)
    # docs_all_words = {}
    print ('size of texts1: {}'.format(sys.getsizeof(texts1)))
    print ('size of texts1: {}'.format(texts1.__sizeof__()))
    print ('size of texts1: {}'.format(size_sum))
    print ('size of id1s: {}'.format(sys.getsizeof(id1s)))
    print ('size of id2s: {}'.format(sys.getsizeof(id2s)))
    print ('size of labels: {}'.format(sys.getsizeof(labels)))
    print ('size of file: {}'.format(sys.getsizeof(file)))
    print ('size of qes_all_words: {}'.format(sys.getsizeof(qes_all_words)))
    print ('size of docs_all_words: {}'.format(sys.getsizeof(docs_all_words)))
    gc.collect()
    print ('middle 2 in {}'.format(mode))
    print('----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # todo: qes_all_words, docs_all_words
    # for i in range(len(docs)):
    #     docs_all_words[docs[i][0]] = docs[i][1]
    texts2 = []
    print ('middle 3 in {}'.format(mode))
    counter = 0
    for t_ in id2s:
        counter +=1
        print (counter)
        print('----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # ld = docs_all_words[t_].split(' ')
        ld = docs_all_words[t_]
        # if len(ld) > doc_fit_max_len:
        #     ld = ld[0:doc_fit_max_len]
        texts2.append(ld)
    # if mode=='test':
    #     return texts1, texts2
    # else:
    # labels=list(file['label'])
    print ('middle 4 in {}'.format(mode))

    labels = to_categorical(numpy.array(labels), num_classes=5)
    #        label_smooth=0.1
    #        labels=labels.clip(label_smooth/2., 1.-label_smooth)
    print ('end in {}'.format(mode))
    # print (gc.get_objects())
    # gc.collect()
    print ('successssssful!')
    return labels, texts1, texts2



def get_texts(file_path, mode=None):
    #    qes = pd.read_csv(question_path, sep='	', dtype=str)
    #    qes = qes.dropna()
    #    file = pd.read_csv(file_path, sep=' ', dtype=str)
    #    file = file.dropna()
    #    q1id, q2id = file['q1'], file['q2']
    #    id1s, id2s = get_ids(q1id), get_ids(q2id)
    file = codecs.open(file_path, 'r', encoding='utf8')
    # todo: qes_all_words, docs_all_words
    # q_file = codecs.open(question_path, 'r', encoding='utf8')
    # d_file = codecs.open(doc_path, 'r', encoding='utf8')

    labels, id1s, id2s = get(file)
    # todo: qes_all_words, docs_all_words
    # qes = getq(q_file)
    # qes_all_words = {}
    # for i in range(len(qes)):
    #     qes_all_words[qes[i][0]] = qes[i][1]
    texts1 = []
    # print (mode)
    # print (len(id1s))
    print ('middle 1 in {}'.format(mode))

    print('4-----ssss-------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    counter = 0
    size_sum = 0
    for t_ in id1s:
        counter +=1
        print (counter)
        print('----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # l = qes_all_words[t_].split(' ')
        l = qes_all_words[t_]
        size_sum += sys.getsizeof(l)
        # if len(l) > query_fit_max_len:
        #     l = l[0:query_fit_max_len]
        texts1.append(l)

    # todo: qes_all_words, docs_all_words
    # docs = getq(d_file)
    # docs_all_words = {}
    print ('size of texts1: {}'.format(sys.getsizeof(texts1)))
    print ('size of texts1: {}'.format(texts1.__sizeof__()))
    print ('size of texts1: {}'.format(size_sum))
    print ('size of id1s: {}'.format(sys.getsizeof(id1s)))
    print ('size of id2s: {}'.format(sys.getsizeof(id2s)))
    print ('size of labels: {}'.format(sys.getsizeof(labels)))
    print ('size of file: {}'.format(sys.getsizeof(file)))
    print ('size of qes_all_words: {}'.format(sys.getsizeof(qes_all_words)))
    print ('size of docs_all_words: {}'.format(sys.getsizeof(docs_all_words)))
    gc.collect()
    print ('middle 2 in {}'.format(mode))
    print('----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # todo: qes_all_words, docs_all_words
    # for i in range(len(docs)):
    #     docs_all_words[docs[i][0]] = docs[i][1]
    texts2 = []
    print ('middle 3 in {}'.format(mode))
    counter = 0
    for t_ in id2s:
        counter +=1
        print (counter)
        print('----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # ld = docs_all_words[t_].split(' ')
        ld = docs_all_words[t_]
        # if len(ld) > doc_fit_max_len:
        #     ld = ld[0:doc_fit_max_len]
        texts2.append(ld)
    # if mode=='test':
    #     return texts1, texts2
    # else:
    # labels=list(file['label'])
    print ('middle 4 in {}'.format(mode))

    labels = to_categorical(numpy.array(labels), num_classes=5)
    #        label_smooth=0.1
    #        labels=labels.clip(label_smooth/2., 1.-label_smooth)
    print ('end in {}'.format(mode))
    # print (gc.get_objects())
    # gc.collect()
    print ('successssssful!')
    return labels, texts1, texts2


# Todo: default is 50
embed_size = 300
num_conv2d_layers = 7
filters_2d = [16, 32, 16, 16, 16, 16, 16]
kernel_size_2d = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
mpool_size_2d = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
dropout_rate = 0.8
batch_size = 64

# Todo
# data_dir = config['encoded_data_dir']
print("Data dir: {}".format(data_dir))
# TRAIN_PATH = 'data/amazon/patio/relation_train.txt'
# TRAIN_PATH = os.path.join(data_dir, 'relation_train.txt')
# VALID_PATH = os.path.join(data_dir, 'relation_valid.txt')
# TEST_PATH = os.path.join(data_dir, 'relation_test.txt')
# # QUESTION_PATH = 'data/amazon/patio/music_inst_user_text.txt'
# QUESTION_PATH = os.path.join(data_dir, 'user_text.txt')
# DOC_PATH = os.path.join(data_dir, 'item_text.txt')
QUESTION_WORD_EMBED = os.path.join(data_dir, 'user_word_embedding.txt')
DOC_WORD_EMBED = os.path.join(data_dir, 'item_word_embedding.txt')
use_embed = True
check_dir = config['checkpoints_dir']
model_path = os.path.join(check_dir, 'model_after_train.h5')
model_path_prime = os.path.join(check_dir, 'mp_new.h5')
model_weight_path = os.path.join(check_dir, 'model_weights.h5')
log_path = 'checkpoints/mp_lrs.txt'
sub_path = 'submission_mp_lrs.csv'

# TRAIN_PATH = 'data/qobatch_sizeuraqp/relation_train.txt'
# VALID_PATH = 'data/qouraqp/relation_valid.txt'
# TEST_PATH = 'data/qouraqp/relation_test.txt'
# QUESTION_PATH = 'data/qouraqp/corpus_preprocessed.txt'
# WORD_EMBED='data/qouraqp/embed_glove_d50'
# use_embed=False
# model_path='checkpoints/mp_lrs.h5'
# log_path='checkpoints/mp_lrs.txt'
# sub_path='submission_mp_lrs.csv'


# # Todo
print("Started...")
start_time = time.time()

print('Load files...')
train_labels, train_texts1, train_texts2 = get_texts(TRAIN_PATH, 'train')
# train_labels, train_texts1, train_texts2 = get_texts(QUESTION_PATH, TRAIN_PATH, 'train')
valid_labels, valid_texts1, valid_texts2 = get_texts(VALID_PATH, 'valid')
# valid_labels, valid_texts1, valid_texts2 = get_texts(QUESTION_PATH, VALID_PATH, 'valid')
print ('pppppppppppppppppppppppppppppppp')
print (len(valid_texts1))
# test_labels, test_texts1, test_texts2 = get_texts(QUESTION_PATH, TEST_PATH, 'test')
test_labels, test_texts1, test_texts2 = get_texts(TEST_PATH, 'test')

# sparse
# u_test1_labels, u_test1_texts1, u_test1_texts2 = get_texts_sparse(TEST_PATH, sparse_user_dict[1], 0, 'test')
# u_test2_labels, u_test2_texts1, u_test2_texts2 = get_texts_sparse(TEST_PATH, sparse_user_dict[2], 0, 'test')
# u_test3_labels, u_test3_texts1, u_test3_texts2 = get_texts_sparse(TEST_PATH, sparse_user_dict[3], 0, 'test')
# u_test4_labels, u_test4_texts1, u_test4_texts2 = get_texts_sparse(TEST_PATH, sparse_user_dict[4], 0, 'test')
# u_test5_labels, u_test5_texts1, u_test5_texts2 = get_texts_sparse(TEST_PATH, sparse_user_dict[5], 0, 'test')
#
#
# i_test1_labels, i_test1_texts1, i_test1_texts2 = get_texts_sparse(TEST_PATH, sparse_item_dict[1], 1, 'test')
# i_test2_labels, i_test2_texts1, i_test2_texts2 = get_texts_sparse(TEST_PATH, sparse_item_dict[2], 1, 'test')
# i_test3_labels, i_test3_texts1, i_test3_texts2 = get_texts_sparse(TEST_PATH, sparse_item_dict[3], 1, 'test')
# i_test4_labels, i_test4_texts1, i_test4_texts2 = get_texts_sparse(TEST_PATH, sparse_item_dict[4], 1, 'test')
# i_test5_labels, i_test5_texts1, i_test5_texts2 = get_texts_sparse(TEST_PATH, sparse_item_dict[5], 1, 'test')




# import pandas
# series = pandas.Series(train_texts1)
# train_texts1 = series.str.split(' ')
# print ('))))))))))))))))))))))))))))))))')
# print (train_texts1.shape)
# print(' (1_1) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# padded_train_texts1=pad_sequences(train_texts1, maxlen=query_fit_max_len, padding='post')
# del train_texts1
# gc.collect()
#
#
# print(' (2) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# series = pandas.Series(train_texts2)
# train_texts2 = series.str.split(' ')
# print ('))))))))))))))))))))))))))))))))')
# print (train_texts2.shape)
#
# print(' (2_1) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# padded_train_texts2=pad_sequences(train_texts2, maxlen=doc_fit_max_len, padding='post')
#
#
# print(' (3) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
#
# series = pandas.Series(valid_texts1)
# valid_texts1 = series.str.split(' ')
# print ('))))))))))))))))))))))))))))))))')
# print (valid_texts1.shape)
#
# print(' (4) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
#
# series = pandas.Series(valid_texts2)
# valid_texts2 = series.str.split(' ')
# print ('))))))))))))))))))))))))))))))))')
# print (valid_texts2.shape)
#
# print(' (5) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# series = pandas.Series(test_texts1)
# test_texts1 = series.str.split(' ')
# print ('))))))))))))))))))))))))))))))))')
# print (test_texts1.shape)
#
# print(' (6) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# series = pandas.Series(test_texts2)
# test_texts2 = series.str.split(' ')
# print ('))))))))))))))))))))))))))))))))')
# print (test_texts2.shape)
#
# print(' (7) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#######################################################
def make_list(l_texts1):
    # print (len(l_texts1))
    l_train_texts1 = []
    i_count = 0
    for str in l_texts1:
        i_count += 1
        # print ('making list: ', i_count)
        # a = numpy.char.split(str, sep=' ')
        # print (len(a))
        a = str.split(' ')
        # print (len(a))
        l_train_texts1.append(a)
        # print('make_list----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return l_train_texts1
#         # l_train_texts1.append(numpy.char.split(str, sep=' '))
#
# print(' (1) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# train_texts1 = make_list(train_texts1, 'train1')
# train_texts2 = make_list(train_texts2, 'train2')
# print(' (2) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# valid_texts1 = make_list(valid_texts1, 'valid1')
# valid_texts2 = make_list(valid_texts2, 'valid2')
# print(' (3) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#
# test_texts1 = make_list(test_texts1, 'test1')
# test_texts2 = make_list(test_texts2, 'test2')
# print(' (4) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# train_texts1 = [numpy.char.split(i, sep=' ') for i in train_texts1]
# train_texts2 = [numpy.char.split(i, sep=' ') for i in train_texts2]


# valid_texts1 = [numpy.char.split(i, sep=' ') for i in valid_texts1]
# valid_texts2 = [numpy.char.split(i, sep=' ') for i in valid_texts2]


# test_texts1 = [numpy.char.split(i, sep=' ') for i in test_texts1]
# test_texts2 = [numpy.char.split(i, sep=' ') for i in test_texts2]


print('Prepare word embedding...')
# # Todo
# pad the docs
# print ('train_texts1.shape')
# print (train_texts1.shape)
# padded_train_texts1=pad_sequences(train_texts1, maxlen=query_fit_max_len, padding='post')
# padded_train_texts2=pad_sequences(train_texts2, maxlen=doc_fit_max_len, padding='post')
# padded_valid_texts1=pad_sequences(valid_texts1, maxlen=query_fit_max_len, padding='post')
# print ('chitoooooo')
# print (type((valid_texts1)))
# print (len(valid_texts1))
# print (padded_valid_texts1)
# padded_valid_texts2=pad_sequences(valid_texts2, maxlen=doc_fit_max_len, padding='post')
# padded_test_texts1=pad_sequences(test_texts1, maxlen=query_fit_max_len, padding='post')
# padded_test_texts2=pad_sequences(test_texts2, maxlen=doc_fit_max_len, padding='post')
# create a weight matrix for words in training docs
user_word_embedding = pandas.read_csv(QUESTION_WORD_EMBED, header=None, sep=' ')
print('user worrrd embedding')
print(user_word_embedding)
user_embedding_matrix = user_word_embedding.iloc[:, 1:-1]
print('user embedding mat')
print(user_embedding_matrix)

item_word_embedding = pandas.read_csv(DOC_WORD_EMBED, header=None, sep=' ')
print('item worrrd embedding')
print(item_word_embedding)
item_embedding_matrix = item_word_embedding.iloc[:, 1:-1]
print('item embedding mat')
print(item_embedding_matrix)

user_vocab_size = len(user_word_embedding)
item_vocab_size = len(item_word_embedding)

def map_lables(labels):
    mapped_labels = numpy.zeros((len(labels), 1))
    for i in range(len(labels)):
        index, value = max(enumerate(labels[i]), key=operator.itemgetter(1))
        mapped_labels[i, 0] = index + 1

    print('map labels:')
    print (type(labels))
    print (labels)
    print (type(mapped_labels))
    print (mapped_labels)

    return mapped_labels


# # Todo
print('Split train and valid set...')
# train_labels, train_texts1, train_texts2 = train_labels, padded_train_texts1, padded_train_texts2
# valid_labels, valid_texts1, valid_texts2 = valid_labels, padded_valid_texts1, padded_valid_texts2
# test_labels, test_texts1, test_texts2 = test_labels, padded_test_texts1, padded_test_texts2

train_labels = map_lables(train_labels)
valid_labels = map_lables(valid_labels)
test_labels = map_lables(test_labels)

# sparse
# u_test1_labels = map_lables(u_test1_labels)
# u_test2_labels = map_lables(u_test2_labels)
# u_test3_labels = map_lables(u_test3_labels)
# u_test4_labels = map_lables(u_test4_labels)
# u_test5_labels = map_lables(u_test5_labels)
#
#
# i_test1_labels = map_lables(i_test1_labels)
# i_test2_labels = map_lables(i_test2_labels)
# i_test3_labels = map_lables(i_test3_labels)
# i_test4_labels = map_lables(i_test4_labels)
# i_test5_labels = map_lables(i_test5_labels)



def mean_sq_error(y_true, y_pred):
    import keras.backend as K
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)

    return K.mean(K.square(y_pred - y_true))


def custom_activation(x):
    import keras.backend as K
    return (K.sigmoid(x) * 4) + 1

def build_model(layer1_dot):
    layer1_conv = Conv2D(filters=8, kernel_size=5, padding='same')(layer1_dot)
    layer1_activation = Activation('relu')(layer1_conv)
    z = MaxPooling2D(pool_size=(2, 2))(layer1_activation)

    for i in range(num_conv2d_layers):
        z = Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
        z = Activation('relu')(z)
        z = MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)

    pool1_flat = Flatten()(z)
    pool_drop = Dropout(rate=dropout_rate)(pool1_flat)
    out = Dense(1, kernel_initializer='random_uniform', activation='linear')(pool_drop)
    return out

def build_model_2(layer1_dot):

    for i in range(num_conv2d_layers):
        if i==0:
            z = layer1_dot
        z = Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
        z = Activation('relu')(z)
        z = BatchNormalization(axis=-1)(z)
        z = MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)

    z = Flatten()(z)
    z = Dense(16)(z)
    z = Activation("relu")(z)
    z = BatchNormalization(axis=-1)(z)
    z = Dropout(dropout_rate)(z)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    z = Dense(4)(z)
    z = Activation("relu")(z)

    z = Dense(1, activation="linear")(z)

    return z


# Todo: Build model begin
print('Build model...')

query=Input(shape=(query_fit_max_len,), name='query')
doc=Input(shape=(doc_fit_max_len,), name='doc')

if use_embed:
    print (user_embedding_matrix.shape)
    print (item_embedding_matrix.shape)
    print ('using___________enmbeddings')
    q_embed=Embedding(user_vocab_size, embed_size, weights=[user_embedding_matrix], trainable=True)(query)
    d_embed=Embedding(item_vocab_size, embed_size, weights=[item_embedding_matrix], trainable=True)(doc)
else:
    q_embed=Embedding(user_vocab_size, embed_size, embeddings_initializer='uniform', trainable=True)(query)
    d_embed=Embedding(item_vocab_size, embed_size, embeddings_initializer='uniform', trainable=True)(doc)

layer1_dot=dot([q_embed, d_embed], axes=-1)
layer1_dot=Reshape((query_fit_max_len, doc_fit_max_len, -1))(layer1_dot)

out = build_model_2(layer1_dot)

# ‌Todo: building the model
# layer1_conv=Conv2D(filters=8, kernel_size=5, padding='same')(layer1_dot)
# layer1_activation=Activation('relu')(layer1_conv)
# z=MaxPooling2D(pool_size=(2,2))(layer1_activation)
#
# for i in range(num_conv2d_layers):
#     z=Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
#     z=Activation('relu')(z)
#     z=MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)
#
# pool1_flat=Flatten()(z)
# pool_drop=Dropout(rate=dropout_rate)(pool1_flat)
# out = Dense(1, activation='linear')(pool_drop)

# pool1_flat=Flatten()(z)
# pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
# # mlp1=Dense(32)(pool1_flat_drop)
# # out=Dense(1, kernel_initializer='normal')(mlp1)
#
#
# mlp1=Dense(32)(pool1_flat_drop)
# mlp1=Activation('relu')(mlp1)
# out=Dense(5, activation='softmax')(mlp1)
# # out = Dense(1, kernel_initializer='normal')(out)

model=Model(inputs=[query, doc], outputs=out)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=5, min_lr=0.0005)

# keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# adagrad = keras.optimizers.Adagrad(lr=0.026309854371732944)
adam = keras.optimizers.Adam(lr=0.001)

# TODO: categorical_crossentropy
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse', 'mae'])
print('metrics: {}'.format(model.metrics_names))
model.summary()

# Todo: Build model end


# build dataset generator
def generator(texts1, texts2, labels, batch_size, min_index, max_index):
    i = min_index

    past = time.time()
    while True:
        if i + batch_size >= max_index:
            i = min_index
        # rows = numpy.arange(i, min(i + batch_size, max_index))
        i += batch_size

        samples1 = texts1[i:min(i + batch_size, max_index)]
        samples1 = make_list(samples1)
        samples1 = pad_sequences(samples1, maxlen=query_fit_max_len, padding='post')


        samples2 = texts2[i:min(i + batch_size, max_index)]
        samples2 = make_list(samples2)
        samples2 = pad_sequences(samples2, maxlen=doc_fit_max_len, padding='post')

        targets = labels[i:min(i + batch_size, max_index)]
        # print('GEN----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Batch training time is {}'.format(time.time() - past))
        past = time.time()
        yield {'query': samples1, 'doc': samples2}, targets


def test_generator(texts1, texts2, batch_size, min_index, max_index):
    i = min_index

    while True:
        if i + batch_size >= max_index:
            i = min_index
        # rows = numpy.arange(i, min(i + batch_size, max_index))
        i += batch_size

        # samples1 = texts1[rows]
        # samples2 = texts2[rows]

        samples1 = texts1[i:min(i + batch_size, max_index)]
        samples1 = make_list(samples1)
        samples1 = pad_sequences(samples1, maxlen=query_fit_max_len, padding='post')


        samples2 = texts2[i:min(i + batch_size, max_index)]
        samples2 = make_list(samples2)
        samples2 = pad_sequences(samples2, maxlen=doc_fit_max_len, padding='post')

        yield {'query': samples1, 'doc': samples2}


# Todo: simple training begin
train_gen=generator(train_texts1, train_texts2, train_labels, batch_size=batch_size, min_index=0, max_index=len(train_texts1))
print(' (2) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

valid_gen=generator(valid_texts1, valid_texts2, valid_labels, batch_size=batch_size, min_index=0, max_index=len(valid_texts1))
print(' (3) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

print ('injaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
test_gen=test_generator(test_texts1, test_texts2, batch_size=1, min_index=0, max_index=len(test_texts1))
test_gen_targets = generator(test_texts1, test_texts2, test_labels, batch_size=1, min_index=0, max_index=len(test_texts1))
# u_test1_labels, u_test1_texts1, u_test1_texts2

# sparse
# u_test1_gen_targets = generator(u_test1_texts1, u_test1_texts2, u_test1_labels, batch_size=1, min_index=0, max_index=len(u_test1_texts1))
# u_test2_gen_targets = generator(u_test2_texts1, u_test2_texts2, u_test2_labels, batch_size=1, min_index=0, max_index=len(u_test2_texts1))
# u_test3_gen_targets = generator(u_test3_texts1, u_test3_texts2, u_test3_labels, batch_size=1, min_index=0, max_index=len(u_test3_texts1))
# u_test4_gen_targets = generator(u_test4_texts1, u_test4_texts2, u_test4_labels, batch_size=1, min_index=0, max_index=len(u_test4_texts1))
# u_test5_gen_targets = generator(u_test5_texts1, u_test5_texts2, u_test5_labels, batch_size=1, min_index=0, max_index=len(u_test5_texts1))
#
# i_test1_gen_targets = generator(i_test1_texts1, i_test1_texts2, i_test1_labels, batch_size=1, min_index=0, max_index=len(i_test1_texts1))
# i_test2_gen_targets = generator(i_test2_texts1, i_test2_texts2, i_test2_labels, batch_size=1, min_index=0, max_index=len(i_test2_texts1))
# i_test3_gen_targets = generator(i_test3_texts1, i_test3_texts2, i_test3_labels, batch_size=1, min_index=0, max_index=len(i_test3_texts1))
# i_test4_gen_targets = generator(i_test4_texts1, i_test4_texts2, i_test4_labels, batch_size=1, min_index=0, max_index=len(i_test4_texts1))
# i_test5_gen_targets = generator(i_test5_texts1, i_test5_texts2, i_test5_labels, batch_size=1, min_index=0, max_index=len(i_test5_texts1))


# s_model = KerasClassifier(build_fn=create_model, verbose=0)
# s_optimizer = ['Adagrad', 'Adam']
# s_learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# s_batch_size = [10, 20, 40, 60, 80, 100]
#
# param_grid = dict(learn_rate = s_learn_rate, batch_size=s_batch_size)
# grid = GridSearchCV(estimator=s_model, param_grid=param_grid, n_jobs=-1)
#
# print (train_texts1.shape, train_texts2.shape, train_labels.shape)
# grid_result = grid.fit(X=train_texts1, y=train_labels)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

print('Train classifier...')
print ('injaaaaaaaaaa')
print (len(valid_texts1))
print (batch_size)
print (len(valid_texts1)//batch_size)

print ('train injaaaaaaaaaa')
print (len(train_texts1))
print (batch_size)
print (len(train_texts1)//batch_size)


start_train = time.time()
print ('loading model')
# model = load_model(model_path)
# model.load_weights(model_weight_path)
print ('model losssgaded!')
history=model.fit_generator(train_gen, epochs=40, steps_per_epoch=len(train_texts1)//batch_size,
                  validation_data=valid_gen, validation_steps=len(valid_texts1)//batch_size, verbose=1,
                  callbacks=[reduce_lr, ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True),
#                             EarlyStopping(monitor='val_loss', patience=3),
                             CSVLogger(log_path)])
print(' (4) ----------------------->Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

end_train = time.time()
print ('saving model weights')
model.save_weights(model_weight_path)
print ('model weights saved!')
# model.save(model_path_prime)

# Todo: simple training end

# history=model.fit_generator(train_gen, epochs=10, steps_per_epoch=len(train_texts1)//batch_size,
#                   verbose=1,
#                   callbacks=[ModelCheckpoint(model_path, monitor='loss', mode='min', save_best_only=True),
# #                             EarlyStopping(monitor='val_loss', patience=3),
#                              CSVLogger(log_path)
#                              ])

# model=load_model(model_path_prime)

# u_test1_gen_targets
# sparse
# print("Evalutation of sparse USER with 1 review:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(u_test1_gen_targets, steps=len(u_test1_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse USER with 2 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(u_test2_gen_targets, steps=len(u_test2_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse USER with 3 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(u_test3_gen_targets, steps=len(u_test3_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse USER with 4 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(u_test4_gen_targets, steps=len(u_test4_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse USER with 5 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(u_test5_gen_targets, steps=len(u_test5_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
#
#
# print("Evalutation of sparse ITEM with 1 review:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(i_test1_gen_targets, steps=len(i_test1_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse ITEM with 2 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(i_test2_gen_targets, steps=len(i_test2_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse ITEM with 3 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(i_test3_gen_targets, steps=len(i_test3_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse ITEM with 4 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(i_test4_gen_targets, steps=len(i_test4_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
#
# print("Evalutation of sparse ITEM with 5 reviews:")
# print('metrics: {}'.format(model.metrics_names))
# print(model.evaluate_generator(i_test5_gen_targets, steps=len(i_test5_texts1)))
# print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

print("############################# Final Evaluation ###################################")
print("Evalutation of best performing model:")
print('metrics: {}'.format(model.metrics_names))
test_start = time.time()
print(model.evaluate_generator(test_gen_targets, steps=len(test_texts1)))
print("############################# TEST TIME FOR ONE SAMPLE ###################################")
print ((time.time()-test_start)/len(test_texts1))

# Todo: simple predicting begin
# print('Predict...')
# # model=load_model(model_path_prime)
preds=model.predict_generator(test_gen, steps=len(test_texts1))
print (preds)
print (preds.shape)
# #
# mse_file = codecs.open(os.path.join(check_dir, 'mse.txt'), 'w', encoding='utf8')
# rating_file = codecs.open(os.path.join(check_dir, 'ratings.txt'), 'w', encoding='utf8')
# rating_file.write('true ratings\t\t\t predicted ratings\n')
#
# print ('predictions')
# print (len(preds))
# for ii in range(len(preds)):
#     # index, value = max(enumerate(preds[ii]), key=operator.itemgetter(1))
#     # if test_labels[ii][index] == 1:
#     #     rating_file.write('1 ')
#     # else:
#     #     rating_file.write('0 ')
#
#
#     rating_file.write('[ ')
#     for r in test_labels[ii]:
#         rating_file.write('{} '.format(r))
#     rating_file.write(']  ')
#
#     rating_file.write('[ ')
#     for rp in preds[ii]:
#         rating_file.write('{} '.format(rp))
#     rating_file.write(']\n')
#
#     # print ('comparing pred with test')
#     # print (preds[ii])
#     # print (test_labels[ii])
#     # print ('//')
#
#
#
# # pos = 0
# # pred_dict = {}
# # ssum = 0
# # for i in range(len(preds)):
# #     index, value = max(enumerate(preds[i]), key=operator.itemgetter(1))
# #     if index not in pred_dict:
# #         pred_dict[index] = 0
# #     pred_dict[index] += 1
# #     if test_labels[i][index] == 1:
# #         pos += 1
# #     real_index, real_value = max(enumerate(test_labels[i]), key=operator.itemgetter(1))
# #     ssum += pow((real_index - index), 2)
# #
# #
# #
# # print ('***********Accuracy***********:')
# # print (pos/len(preds))
# # mse_file.write('***********Accuracy***********:')
# # mse_file.write(str(pos/len(preds)))
# #
# # print ('***********MSE***********:')
# # print (ssum/len(preds))
# # mse_file.write('***********MSE***********:')
# # mse_file.write(str(ssum/len(preds)))
# #
# # import keras.backend as K
# #
# # y_true = K.argmax(test_labels)
# # y_pred = K.argmax(preds)
# # qq = 0
# # y_true = keras.backend.get_value(y_true)
# # y_pred = keras.backend.get_value(y_pred)
# #
# # for i,j in zip(y_true, y_pred):
# #     qq += pow(i-j,2)
# #
# # print ('$$$')
# # print (qq/len(preds))
# #
# # print ('$$$')
# # print (keras.backend.get_value(keras.losses.mean_squared_error(y_true, y_pred)))
# #
# # # print ('***********Accuracy***********:')
# # # acc_value = keras.metrics.categorical_accuracy(test_labels, preds)
# # # print (acc_value)
# #
# # print ('**********************:')
# # print (pred_dict[4])
# # Todo: simple predicting end
#
# # print ("keras.losses.mean_squared_error")
# #
# # print ( 'keras.losses.mean_squared_error: {}'.format(keras.backend.get_value(keras.losses.mean_squared_error(test_labels, preds))))
#
#
#
# # Todo: Plot validation accuracy and loss...
# print('Plot validation accuracy and loss...')
# print('Plot accuracy and loss...')
# import matplotlib.pyplot as plt
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
#
#
# mse = history.history['mean_squared_error']
# val_mse = history.history['val_mean_squared_error']
#
# # cat_acc = history.history['categorical_accuracy']
# # val_cat_acc = history.history['val_categorical_accuracy']
# #
# # acc = history.history['acc']
# # val_acc = history.history['val_acc']
#
# mae = history.history['mean_absolute_error']
# val_mae = history.history['val_mean_absolute_error']
#
# plt.plot(loss, label='loss')
# plt.plot(val_loss, label='val_loss')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
# plt.show()
#
# plt.plot(mse, label='mse')
# plt.plot(val_mse, label='val_mse')
# plt.title('model mse')
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
# plt.show()
#
# # plt.plot(cat_acc, label='cat_acc')
# # plt.plot(val_cat_acc, label='val_cat_acc')
# # plt.title('model categorical accuracy')
# # plt.ylabel('categorical accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'valid'], loc='upper left')
# # plt.show()
# #
# # plt.plot(acc, label='acc')
# # plt.plot(val_acc, label='val_acc')
# # plt.title('model accuracy')
# # plt.ylabel('accuracy')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'valid'], loc='upper left')
# # plt.show()
#
# plt.plot(mae, label='mae')
# plt.plot(val_mae, label='val_mae')
# plt.title('model mae')
# plt.ylabel('mae')
# plt.xlabel('epoch')
# plt.legend(['train', 'valid'], loc='upper left')
# plt.show()

# for m in mse:
#     mse_file.write('{}\n'.format(m))
#     # print (m)
# print ('type(mse)')
# print (type(mse))

print('train took {} minutes.'.format(((end_train - start_train)/60)))

end_time = time.time()
print('Fin! took {} minutes.'.format(((end_time - start_time) / 60)))
