import sys
import yaml
from gensim.models.keyedvectors import KeyedVectors
#from gensim.models import KeyedVectors
import json
import numpy as np

class Data():
    def __init__(self, data):
        self.sentence = data[0]
        self.label = data[1]
        self.entity_position = data[2]
        self.drug_name = data[3]

    def make_develop_data(self):
        statistics = [self.label.count(i) for i in range(5)]
        ratio = 4   # n_train : n_dev = ratio : 1
        for x in self.label:
            pass
            #print(x)

    def make_validation_data(self, n_fold):
        pass
        

    def preprocess_train_data(self, word_vec_path):
        w2v_model = KeyedVectors.load_word2vec_format(word_vec_path, binary=True)
        # Initialize vectors of entities with vector of 'drug'
        if True:
            #entity_name = ['ENTITY1', 'ENTITY2', 'ENTITYOTHER']
            entity_name = ['ENTITY1', 'ENTITY2']
            for x in entity_name:
                w2v_model.vocab[x.lower()] = w2v_model.vocab['drug']

        freq_dict = {}
        index_dict = {}
        word_vectors = []

        # Make word vectors (from Pretrained dataset)
        for i, key in enumerate(w2v_model.vocab.keys()):
            index_dict[key] = i
            word_vectors.append(w2v_model.wv[key])
        #print(len(word_vectors))

        # Count frequencies of words in Train dataset
        for sent in self.sentence:
            for word in sent:
                word = word.lower()
                if word not in freq_dict:
                    freq_dict[word] = 1
                else:
                    freq_dict[word] += 1

        index_array = np.zeros([len(sent_tr), args.max_sent_len], dtype='i')
        label_array = np.array(label_tr)
        ep_tr = np.array(entity_pos_tr)
        mask_tr = np.zeros([len(sent_tr), args.max_sent_len], dtype='f')
        sent_len_tr = np.zeros([len(sent_tr)], dtype='i')
        # Word -> index (Train dataset)
        # Add new words in Train dataset to index
        for i, sent in enumerate(sent_tr):
            for j, word in enumerate(sent):
                word = word.lower()
                if freq[word] > 1:
                    if word not in index:
                        n_new_train += 1
                        index[word] = idx_low_freq + n_new_train
                    index_tr[i][j] = index[word]
                else:
                    index_tr[i][j] = idx_low_freq
            mask_tr[i][:len(sent)] = 1

        # Initialize vectors of low_freq words and UNK words
        # with mean of pretrained word vectors
        unk_vector = np.mean(np.array(word_vectors), axis=0).tolist()
        for i in range(1 + n_new_train):
            word_vectors.append(unk_vector)

        w_v = np.array(word_vectors)

        self.freq = freq
        self.index = index
        self.word_vectors = word_vectors

    def preprocess_test_data(self, train_data):
        pass

def main(path_tr, path_te, args):
    model = KeyedVectors.load_word2vec_format('../../W2V/w2v_PubMed2014_min10.bin', binary=True)

    # Initialize vectors of entities with vector of 'drug'
    if True:
        #entity_name = ['ENTITY1', 'ENTITY2', 'ENTITYOTHER']
        entity_name = ['ENTITY1', 'ENTITY2']
        for x in entity_name:
            model.vocab[x.lower()] = model.vocab['drug']

    with open(path_tr, mode='r') as f_tr:
        sent_tr, label_tr, entity_pos_tr, name_tr, _ = json.load(f_tr)
    with open(path_te, mode='r') as f_te:
        sent_te, label_te, entity_pos_te, name_te, _ = json.load(f_te)

    freq = {}
    index = {}
    word_vectors = []

    # Make word vectors (from Pretrained dataset)
    for i, key in enumerate(model.vocab.keys()):
        index[key] = i
        word_vectors.append(model.wv[key])
    #print(len(word_vectors))

    # Count frequencies of words in Train dataset
    for sent in sent_tr:
        for word in sent:
            word = word.lower()
            if word not in freq:
                freq[word] = 1
            else:
                freq[word] += 1

    idx_low_freq = len(index)   # index of low frequency words
    n_new_train = 0             # number of New words in Train dataset

    index_tr = np.zeros([len(sent_tr), args.max_sent_len], dtype='i')
    label_tr = np.array(label_tr)
    ep_tr = np.array(entity_pos_tr)
    mask_tr = np.zeros([len(sent_tr), args.max_sent_len], dtype='f')
    sent_len_tr = np.zeros([len(sent_tr)], dtype='i')
    # Word -> index (Train dataset)
    # Add new words in Train dataset to index
    for i, sent in enumerate(sent_tr):
        for j, word in enumerate(sent):
            word = word.lower()
            if freq[word] > 1:
                if word not in index:
                    n_new_train += 1
                    index[word] = idx_low_freq + n_new_train
                index_tr[i][j] = index[word]
            else:
                index_tr[i][j] = idx_low_freq
        mask_tr[i][:len(sent)] = 1

    # Initialize vectors of low_freq words and UNK words
    # with mean of pretrained word vectors
    unk_vector = np.mean(np.array(word_vectors), axis=0).tolist()
    for i in range(1 + n_new_train):
        word_vectors.append(unk_vector)

    w_v = np.array(word_vectors)

    index_te = np.zeros([len(sent_te), args.max_sent_len], dtype='i')
    label_te = np.array(label_te)
    ep_te = np.array(entity_pos_te)
    mask_te = np.zeros([len(sent_te), args.max_sent_len], dtype='f')
    sent_len_te = np.zeros([len(sent_te)], dtype='i')
    # Word -> index (Test dataset)
    # UNK word index is same as low frequency word index
    for i, sent in enumerate(sent_te):
        for j, word in enumerate(sent):
            word = word.lower()
            if word in index:
                index_te[i][j] = index[word]
            else:
                index_te[i][j] = idx_low_freq
        mask_te[i][:len(sent)] = 1

    return [index_tr, label_tr, ep_tr, mask_tr, name_tr],\
           [index_te, label_te, ep_te, mask_te, name_te],\
           w_v 
