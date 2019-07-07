import codecs
import json
import math
import operator
import os
import pickle
import sys
from collections import Counter

import gensim
import numpy as np

config = json.loads(open(sys.argv[1]).read())

w2v_model_file = config['w2v_model_file']
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file, binary = True)


def extract_keywords(text, inv_vocab):
    profile_dict = {}
    # overall_dict = Counter()

    for p_id in text:
        profile_dict[p_id] = Counter()
        for w_id in text[p_id]:
            word = inv_vocab[w_id]
            # if word in model.wv.vocab:
            profile_dict[p_id].update([word])

    # if mode == 'corpus':
    #     for p in profile_dict:
    #         overall_dict.update(profile_dict[p])

    return profile_dict


def word_doc_dict(profile_dict):
    words = set()
    for p_id in profile_dict:
        words |= set([w for w in profile_dict[p_id]])

    words_dict = {}
    for w in words:
        for p_id in profile_dict:
            if w in profile_dict[p_id]:
                if w not in words_dict:
                    words_dict[w] = []
                words_dict[w].append(p_id)

    return words_dict

def get_word_idf_dict(profile_dict1, profile_dict2):
    wd_dict1 = word_doc_dict(profile_dict1)
    wd_dict2 = word_doc_dict(profile_dict2)

    words = set()
    words |= set([w for w in wd_dict1])
    words |= set([w for w in wd_dict2])

    N = len(profile_dict1) + len(profile_dict2)

    for word in words:
        l1 = 0
        l2 = 0
        if word in wd_dict1:
            l1 = len(wd_dict1[word])
        if word in wd_dict2:
            l2 = len(wd_dict2[word])


    word_idf_dict = {word: math.log10(N/(l1 + l2)) for word in words}

    return word_idf_dict

def get_word_tf(word, doc):
    word_count = doc[word]
    total = 0
    for w in doc:
        total += doc[w]

    return word_count/total

def get_sorted_encoded_profile_dict(profile_dict1, vocab1, word_idf_dict):
    sorted_profile_encoded_dict1 = {}
    for p_id in profile_dict1:
        word_tf_idf_dict = {}
        for word in profile_dict1[p_id]:
            if word not in word_tf_idf_dict:
                word_tf = get_word_tf(word, profile_dict1[p_id])
                word_idf = word_idf_dict[word]
                word_tf_idf_dict[word] = word_tf * word_idf

        sorted_profile_text = sorted(word_tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_profile_encoded_dict1[p_id] = [vocab1[w] for w,score in sorted_profile_text]

    return sorted_profile_encoded_dict1


def encode_top_tf_idf_words(text1, inv_vocab1, vocab1, text2, inv_vocab2, vocab2):
    print ("TF-IDF")
    profile_dict1 = extract_keywords(text1, inv_vocab1)
    profile_dict2 = extract_keywords(text2, inv_vocab2)

    word_idf_dict = get_word_idf_dict(profile_dict1, profile_dict2)

    sorted_profile_encoded_dict1 = get_sorted_encoded_profile_dict(profile_dict1, vocab1, word_idf_dict)
    sorted_profile_encoded_dict2 = get_sorted_encoded_profile_dict(profile_dict2, vocab2, word_idf_dict)

    return sorted_profile_encoded_dict1, sorted_profile_encoded_dict2



def build_prfile_text_file(profile_text, inv_vocab, file):
    # profile_dict = {}
    # overall_dict = Counter()
    res_text = {}
    for profile_id in profile_text:
        file.write('{}\t'.format(profile_id))
        if profile_id not in res_text:
            res_text[profile_id] = []
        # profile_dict[profile_id] = Counter()
        print ('len(profile_text[profile_id])')
        print (len(profile_text[profile_id]))
        for word_id in profile_text[profile_id]:
            # if inv_vocab[word_id] in model.wv.vocab:
                # if profile_id == 321:
                #     print ('%%%%s')
                #     print(inv_vocab[word_id])
            file.write('{} '.format(word_id))
            res_text[profile_id].append(word_id)

        print('len(res_text[profile_id])')
        print(len(res_text[profile_id]))
        print ('@@@@@@@@@@@@@')

            # profile_dict[profile_id].update([inv_vocab[word_id]])
            # else:
            #     if profile_id == 321:
            #         print ('%%%%not')
            #         print(inv_vocab[word_id])
        file.write('\n')
    # print ('^^^')
    # print (len(res_text[321]))
    # if mode == 'corpus':
    #     for p in profile_dict:
    #         overall_dict.update(profile_dict[p])

    return res_text
        # , profile_dict, overall_dict


# def build_word_embedding_file(inv_vocab, file):
#     for word_id in range(len(inv_vocab)):
#        word = inv_vocab[word_id]
#        if word in model.wv.vocab:
#            file.write('{} '.format(word_id))
#            word_vec = model[word]
#            for e in word_vec:
#                file.write('{} '.format(e))
#            file.write('\n')


def build_word_embedding_file(W, file):
    for idx in range(W.shape[0]):
        file.write('{} '.format(idx))
        for e in W[idx]:
            file.write('{} '.format(e))
        file.write('\n')


def get_word_embedding(vocabulary_user):
    # initial matrix with random uniform
    u = 0
    initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), config['embedding_dim']))
    # load any vectors from the word2vec
    print("Load word2vec u file {}\n".format(w2v_model_file))
    with open(w2v_model_file, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = 0

            if word in vocabulary_user:
                u = u + 1
                idx = vocabulary_user[word]
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return initW


data_processed_dir = config['data_general_processed_dir']

data_image_file = os.path.join(data_processed_dir, '{}_data.image'.format(config['data_name']))
user_text_file = os.path.join(data_processed_dir, 'user.text')
item_text_file = os.path.join(data_processed_dir, 'item.text')
user_vocab_file = os.path.join(data_processed_dir, 'user.vocab')
item_vocab_file = os.path.join(data_processed_dir, 'item.vocab')
user_inv_vocab_file = os.path.join(data_processed_dir, 'user.invvocab')
item_inv_vocab_file = os.path.join(data_processed_dir, 'item.invvocab')
user_vocab_wv_file = os.path.join(data_processed_dir, 'user_vocab.wv')
item_vocab_wv_file = os.path.join(data_processed_dir, 'item_vocab.wv')

print (user_text_file)
user_text_pkl = open(user_text_file, 'rb')
user_text = pickle._Unpickler(user_text_pkl)
user_text.encoding = 'latin1'
user_text = user_text.load()

data_image_pkl = open(data_image_file, 'rb')
data_image = pickle._Unpickler(data_image_pkl)
data_image.encoding = 'latin1'
data_image = data_image.load()

item_text_pkl = open(item_text_file, 'rb')
item_text = pickle._Unpickler(item_text_pkl)
item_text.encoding = 'latin1'
item_text = item_text.load()


user_vocab_pkl = open(user_vocab_file, 'rb')
user_vocab = pickle._Unpickler(user_vocab_pkl)
user_vocab.encoding = 'latin1'
user_vocab = user_vocab.load()
print ('*******')
print (type(user_vocab))


item_vocab_pkl = open(item_vocab_file, 'rb')
item_vocab = pickle._Unpickler(item_vocab_pkl)
item_vocab.encoding = 'latin1'
item_vocab = item_vocab.load()

user_inv_vocab_pkl = open(user_inv_vocab_file, 'rb')
user_inv_vocab = pickle._Unpickler(user_inv_vocab_pkl)
user_inv_vocab.encoding = 'latin1'
user_inv_vocab = user_inv_vocab.load()

item_inv_vocab_pkl = open(item_inv_vocab_file, 'rb')
item_inv_vocab = pickle._Unpickler(item_inv_vocab_pkl)
item_inv_vocab.encoding = 'latin1'
item_inv_vocab = item_inv_vocab.load()

user_vocab_wv_pkl = open(user_vocab_wv_file, 'rb')
user_vocab_wv = pickle._Unpickler(user_vocab_wv_pkl)
user_vocab_wv.encoding = 'latin1'
user_vocab_wv = user_vocab_wv.load()

item_vocab_wv_pkl = open(item_vocab_wv_file, 'rb')
item_vocab_wv = pickle._Unpickler(item_vocab_wv_pkl)
item_vocab_wv.encoding = 'latin1'
item_vocab_wv = item_vocab_wv.load()


# print (data_image, user_text, item_text, user_vocab, item_vocab)
print (len(data_image['train'][0]), len(data_image['valid'][0]), len(data_image['test'][0]), len(user_text), len(item_text), len(user_vocab), len(item_vocab))

data_path = config['encoded_data_dir']

# train_file = codecs.open(data_path + 'relation_train.txt', 'w', encoding='utf8')
train_file = codecs.open(os.path.join(data_path, 'relation_train.txt'), 'w', encoding='utf8')
test_file = codecs.open(os.path.join(data_path, 'relation_test.txt'), 'w', encoding='utf8')
valid_file = codecs.open(os.path.join(data_path, 'relation_valid.txt'), 'w', encoding='utf8')

print('(relation_train), relation_valid, relation_test')
print (train_file, valid_file, test_file)

user_text_file = codecs.open(os.path.join(data_path, 'user_text.txt'), 'w', encoding='utf8')
item_text_file = codecs.open(os.path.join(data_path, 'item_text.txt'), 'w', encoding='utf8')
user_word2vec_file = codecs.open(os.path.join(data_path, 'user_word_embedding.txt'), 'w', encoding='utf8')
item_word2vec_file = codecs.open(os.path.join(data_path, 'item_word_embedding.txt'), 'w', encoding='utf8')
word_dict_file = codecs.open(os.path.join(data_path, 'word_dict.txt'), 'w', encoding='utf8')

train_file.write('label q1 q2\n')
test_file.write('label q1 q2\n')
valid_file.write('label q1 q2\n')
user_text_file.write('qid\twords\n')
item_text_file.write('qid\twords\n')


[uid_train, iid_train, y_train] = data_image['train']
[uid_valid, iid_valid, y_valid] = data_image['valid']
[uid_test, iid_test, y_test] = data_image['test']

print('len(y_train), len(y_valid), len(y_test)')
print (len(y_train), len(y_valid), len(y_test))

for (u, i, l) in zip(uid_train, iid_train, y_train):
    train_file.write('{} {} {}\n'.format(int(float(l)) - 1, u, i))


for (u, i, l) in zip(uid_valid, iid_valid, y_valid):
    valid_file.write('{} {} {}\n'.format(int(float(l)) - 1, u, i))


for (u, i, l) in zip(uid_test, iid_test, y_test):
    test_file.write('{} {} {}\n'.format(int(float(l)) - 1, u, i))


print  (type(user_text))
print  (user_text[1].shape)
s = 0
for w_id in user_text[1]:
    if w_id:
        s+=1
    print (w_id)
    print('***********')
    print (user_inv_vocab[w_id])

print ('***********')
print (s)

# user_text, item_text = encode_top_tf_idf_words(user_text, user_inv_vocab, user_vocab, item_text, item_inv_vocab, item_vocab)

filtered_user_text = build_prfile_text_file(user_text, user_inv_vocab, user_text_file)
filtered_item_text = build_prfile_text_file(item_text, item_inv_vocab, item_text_file)


import matplotlib.pyplot as plt

user_text_lengths = [len(filtered_user_text[user]) for user in filtered_user_text]
item_text_lengths = [len(filtered_item_text[item]) for item in filtered_item_text]

plt.hist(user_text_lengths, normed=True, bins=100)
plt.ylabel('Probability')
plt.savefig(os.path.join(data_path, 'user_text_length.png'))
plt.show()

import statistics
print ('---> mean of user_text_lengths:')
print(statistics.mean(user_text_lengths))

plt.hist(item_text_lengths, normed=True, bins=100)
plt.ylabel('Probability')
plt.savefig(os.path.join(data_path, 'item_text_length.png'))
plt.show()

print ('---> mean of item_text_lengths:')
print(statistics.mean(item_text_lengths))


# build_word_embedding_file(user_inv_vocab, user_word2vec_file)
# build_word_embedding_file(item_inv_vocab, item_word2vec_file)


# user_W = get_word_embedding(user_vocab)
build_word_embedding_file(user_vocab_wv, user_word2vec_file)

# item_W = get_word_embedding(item_vocab)
build_word_embedding_file(item_vocab_wv, item_word2vec_file)

