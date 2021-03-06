#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 18:38
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : BiLSTM-Residual.py
# @Software: PyCharm

import numpy as np
import random
import re
from file_prepare import preprocess, cuu, combine_all, pred2label
from BiLSTM_CRF import plot_scores, generate_datasets
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras_contrib.layers import CRF
from keras_self_attention import SeqSelfAttention
from keras.utils import to_categorical
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


def bilstm_res(train_loc, test_loc):
    train_pre = preprocess(train_loc)
    test_pre = preprocess(test_loc)
    cc_train = cuu(train_pre)
    cc_test = cuu(test_pre)
    words_all, tags_all = combine_all(cc_train, cc_test)
    n_tags = len(tags_all)
    n_words = len(words_all)
    max_len = 130
    word2idx = {w: i for i, w in enumerate(words_all)}
    tag2idx = {t: i for i, t in enumerate(tags_all)}

    X = [[word2idx[w[0]] for w in s] for s in cc_train]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
    X1 = [[word2idx[w[0]] for w in s] for s in cc_test]
    X1 = pad_sequences(maxlen=max_len, sequences=X1, padding="post", value=n_words - 1)

    y = [[tag2idx[w[1]] for w in s] for s in cc_train]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y1 = [[tag2idx[w[1]] for w in s] for s in cc_test]
    y1 = pad_sequences(maxlen=max_len, sequences=y1, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=50,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=250, return_sequences=True,
                               recurrent_dropout=0.2))(model)  # variational biLSTM
    model1 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.2))(model)
    x = add([model, model1])

    attn = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l1(1e-4),
                           attention_regularizer_weight=1e-4,
                           name='Attention')(x)

    model = TimeDistributed(Dense(50, activation="relu"))(x)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)
    model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    history = model.fit(X, np.array(y), batch_size=8, epochs=15, verbose=1)

    test_pred = model.predict(X, verbose=1)
    idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels = pred2label(test_pred, idx2tag)
    true_labels = pred2label(y, idx2tag)

    y1 = [to_categorical(i, num_classes=n_tags) for i in y1]
    test_pred1 = model.predict(X1, verbose=1)
    pred_labels1 = pred2label(test_pred1, idx2tag)
    true_labels1 = pred2label(y1, idx2tag)
    f1 = f1_score(true_labels1, pred_labels1)
    precision = precision_score(true_labels1,pred_labels1)
    recall = recall_score(true_labels1, pred_labels1)
    test_socres = [f1, precision,recall]
    print('Testing ******************')
    print(test_socres)
    return test_socres


def kfold_bilstm_res(dir,d_train,d_test,num_tr,num_te):
    f1 = []
    recall = []
    precision = []
    idx1 = set(range(1, num_tr + 1))
    idx2 = set(range(1, num_te + 1))
    test_num1 = int(num_tr * 0.2)
    test_num2 = int(num_te * 0.2)
    for i in range(5):
        temp1 = random.sample(idx1, test_num1)  # the first 0.2 for testing
        temp2 = random.sample(idx2, test_num2)
        idx1 = idx1 - set(temp1)  # the remaining 0.8 indexes for training
        idx2 = idx2 - set(temp2)
        train_loc = generate_datasets(dir, d_train, num_tr, temp1)[1]
        test_loc = generate_datasets(dir, d_test, num_te, temp2)[0]
        s = bilstm_res(train_loc, test_loc)
        f1.append(s[0])
        recall.append(s[1])
        precision.append(s[2])
    print('kfold mean scores:', np.mean(f1), np.mean(recall), np.mean(precision))
    return np.mean(f1), np.mean(recall), np.mean(precision)


def domain_specific(dir, domains):
    f1_te = []
    precision_te = []
    recall_te = []
    for d in domains:
        if d == 'Overall':
            num_of_files = 110
        else:
            num_of_files = 10
        print('===========', d)
        scores = kfold_bilstm_res(dir, d, d, num_of_files, num_of_files )
        print('==========', d)
        f1_te.append(scores[0])
        precision_te.append(scores[1])
        recall_te.append(scores[2])
    plot_scores(f1_te, precision_te, recall_te, domains, 'BiLSTM-Res-Domain-Specific-Scores(test)')
    f1 = [round(s,2) for s in f1_te]
    recall = [round(s,2) for s in recall_te]
    precision = [round(s,2) for s in precision_te]
    print('Domain-specific:', '\n', 'Testing Scores:', '\n', f1, '\n', precision, '\n', recall)


def f1_domains(dir,domains):
    f1_te = {'Math': [], 'Med': [], 'ES': [], 'Chem': [], 'CS': [], 'Astr': [], 'Agr': [], 'MS': [], 'Bio': [],
             'Eng': []}  # test domain as the key
    for i in range(len(domains) - 1):
        #test_d = generate_datasets(dir, domains[i], 10, random.sample(set(range(1, 11)), 2))[0]
        for j in range(len(domains) - 1):
            #train_d = generate_datasets(dir, domains[i], 10, random.sample(set(range(1, 11)), 2))[1]
            print('============== Trained on {}, Tested on {}'.format(domains[j], domains[i]))
            test_loc = dir + domains[i] + '/test_com.txt'
            train_loc = dir + domains[j] + '/train_com.txt'
            scores = bilstm_res(train_loc, test_loc)
            f1_te[domains[i]].append(scores[0])
    width = 9 / 100
    x = np.arange(10)
    fig, ax = plt.subplots()
    ax.bar(x - 4 * width, f1_te['Math'], width, label='Math')
    ax.bar(x - 3 * width, f1_te['Med'], width, label='Med')
    ax.bar(x - 2 * width, f1_te['ES'], width, label='ES')
    ax.bar(x - width, f1_te['Chem'], width, label='Chem')
    ax.bar(x, f1_te['CS'], width, label='CS')
    ax.bar(x + width, f1_te['Astr'], width, label='Astr')
    ax.bar(x + 2 * width, f1_te['Agr'], width, label='Agr')
    ax.bar(x + 3 * width, f1_te['MS'], width, label='MS')
    ax.bar(x + 4 * width, f1_te['Bio'], width, label='Bio')
    ax.bar(x + 5 * width, f1_te['Eng'], width, label='Eng')
    ax.set_ylabel('Scores')
    ax.set_title('CRF F1 Scores Trained and Tested on Each Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(f1_te.keys())
    ax.legend()
    plt.savefig('/data/xwang/OA-STM-domains/{}'.format('BiLSTM-CRF-per-domain'))
    plt.show()
    for k in f1_te.keys():
        f1_te[k] = [round(s,2) for s in f1_te[k]]
    print('F1 scores:', '\n', f1_te)





if __name__ == '__main__':
    #domains = ['Overall']
    dir = '/data/xwang/OA-STM-domains/'
    #domains = ['Math', 'Med', 'ES', 'Chem', 'CS', 'Astr', 'Agr', 'MS', 'Bio', 'Eng', 'Overall']
    #domain_specific(dir,domains)
    domains = ['Math', 'Med', 'ES', 'Chem', 'CS', 'Astr', 'Agr', 'MS', 'Bio', 'Eng']
    f1_domains(dir,domains)