import numpy as np
import pickle as pkl
from gensim.models import KeyedVectors


def to_indx(base, params, word_indx, label_indx, training):
        
    with open(base + '.split.sent', 'r') as f:
        parsed_sents = f.read().strip().split('\n\n')
    sents = [[x.split('\t')[0] for x in psent.split('\n')] for psent in parsed_sents]
    with open(base + '.label', 'r') as f:
        labels = f.read().strip().split('\n')

    if params['mol2v_path'] is None:
        mol2v_table = None
    else:
        with open(base + '.dbid', 'r') as f:
            dbids = f.read().strip().split('\n')
        with open(params['dbid_dict_path'], 'rb') as f:
            dbid_dict = pkl.load(f)
        mol2v_table = np.load(params['mol2v_path'])
        # Set unknown molecular vector to zero
        dbid_dict['None'] = len(dbid_dict)
        mol2v_table = np.concatenate((mol2v_table, np.zeros((1, mol2v_table.shape[1]))))

    if training:
        # Set paddig term index at 0
        word_indx['<PAD>'] = 0

        if params['w2v_path'] is None:
            w2v_table = np.zeros((2, params['wemb_size']))
        else:
            w2v_model = KeyedVectors.load_word2vec_format(params['w2v_path'], binary=True)

            # Initialize vectors of entities with vector of 'drug'
            for x in ['DRUG1', 'DRUG2']:
                w2v_model.vocab[x.lower()] = w2v_model.vocab['drug']

            w2v_table = np.zeros((len(w2v_model.vocab)+1, w2v_model.vector_size))

            for k in w2v_model.vocab:
                k = k.lower()
                word_indx[k] = len(word_indx)
                w2v_table[word_indx[k]] = w2v_model[k]

        word_freq = {}
        for sent in sents:
            for word in sent:
                word = word.lower()
                # Count freq of words in train data
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

                # Add train data vocab to idex dict
                if word not in word_indx:
                    word_indx[word] = len(word_indx)
        word_indx['<UNK>'] = len(word_indx)

        low_freq_words = [k for k, v in word_freq.items() if v <= params['lfw_threshold']]
        for lfw in low_freq_words:
            word_indx[lfw] = word_indx['<UNK>']
            #word_indx[lfw] = len(word_indx)

        for label in labels:
            if label not in label_indx:
                label_indx[label] = len(label_indx)

        # Initialize new word vectors with average of all pre-trained vectors
        mean_pretrain = np.mean(w2v_table[1:], axis=0)
        new_vec = np.tile(mean_pretrain, (len(word_indx)-len(w2v_table), 1))
        w2v_table = np.concatenate((w2v_table, new_vec), axis=0)

    else:
        w2v_table = None

    # Array input
    max_sent_len = params['max_sent_len']
    #X = np.zeros((len(sents), 3 * max_sent_len)).astype('i')
    X = np.zeros((len(sents), 3*max_sent_len+2)).astype('i')
    for i, sent in enumerate(sents):
        entity_position = [sent.index('DRUG1'), sent.index('DRUG2')]
        for j, word in enumerate(sent):
            word = word.lower()
            # Array word
            if word in word_indx:
                X[i, j] = word_indx[word]
            else:
                X[i, j] = word_indx['<UNK>']
                #X[i, j] = len(word_indx)
            # Array word position from target entity
            X[i, j+max_sent_len] = j - entity_position[0] + max_sent_len
            X[i, j+2*max_sent_len] = j - entity_position[1] + max_sent_len

        if params['mol2v_path'] is not None:
            db1, db2 = dbids[i].split('\t')
            X[i, 3*max_sent_len] = dbid_dict[db1]
            X[i, 3*max_sent_len+1] = dbid_dict[db2]
            
    # Array label
    Y = np.array([label_indx[l] for l in labels]).astype('i')

    return X, Y, w2v_table, mol2v_table
