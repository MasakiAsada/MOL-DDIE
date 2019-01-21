params = {
        'train_path' : '../corpus/semeval2013/instances/train',
        'test_path' : '../corpus/semeval2013/instances/test',
        #'w2v_path' : '../../W2V/w2v_PubMed2014_min10.bin',
        'w2v_path' : None,
        'lfw_threshold' : 1,
        'max_sent_len' : 160,
        'l2_lamda' : 0.0001,
        'learning_rate' : 0.001,
        'batchsize' : 50,
        'n_epoch' : 10,
        'n_filter' : 100,
        'window_size' : [3,5,7],
        'wemb_size' : 200,
        'posemb_size' : 20,
        'posemb_range' : 0.01,
        'mid_dim' : 200,
        'averaging': True
}

