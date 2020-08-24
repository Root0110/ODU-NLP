#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 18:36
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : BiLSTM-E.py
# @Software: PyCharm

import numpy as np
import random
import re
from file_prepare import preprocess, cuu, combine_all, pred2label
from CRF_kfold import generate_datasets, plot_scores

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt


def x_char(cc, max_len, max_len_char, char2idx):
    X_char = []
    for sentence in cc:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    return X_char


def original(y,idx2tag):
    sent_tag_list = []
    for sent_tags in y:
        word_tag_list = []
        for tag in sent_tags:
            word_tag_list.append(idx2tag[tag])
        sent_tag_list.append(word_tag_list)
    for i in range(len(sent_tag_list)):
        for j in range(len(sent_tag_list[i])):
            if sent_tag_list[i][j] == 'PAD':
                sent_tag_list[i][j] = sent_tag_list[i][j].replace('PAD', 'O')
    return sent_tag_list


def bilstm_character(train_loc, test_loc):
    train_pre = preprocess(train_loc)
    test_pre = preprocess(test_loc)
    cc_train = cuu(train_pre)
    cc_test = cuu(test_pre)
    words_all, tags_all = combine_all(cc_train, cc_test)
    n_tags = len(tags_all)
    n_words = len(words_all)

    max_len = 130
    max_len_char = 10

    word2idx = {w: i + 2 for i, w in enumerate(words_all)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    tag2idx = {t: i + 1 for i, t in enumerate(tags_all)}
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}

    X_word = [[word2idx[w[0]] for w in s] for s in cc_train]
    X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

    chars = set([w_i for w in words_all for w_i in w])
    n_chars = len(chars)
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    X_char = x_char(cc_train, max_len, max_len_char, char2idx)

    y = [[tag2idx[w[1]] for w in s] for s in cc_train]
    y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

    # input and embedding for words
    word_in = Input(shape=(max_len,))
    emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                         input_length=max_len, mask_zero=True)(word_in)

    # input and embeddings for characters
    char_in = Input(shape=(max_len, max_len_char,))
    emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                                         input_length=max_len_char, mask_zero=True))(char_in)
    # character LSTM to get word encodings by characters
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                    recurrent_dropout=0.5))(emb_char)

    # main LSTM
    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)
    model = Model([word_in, char_in], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.summary()
    history = model.fit([X_word,
                         np.array(X_char).reshape((len(X_char), max_len, max_len_char))],
                        np.array(y).reshape(len(y), max_len, 1),
                        batch_size=2, epochs=15, validation_split=0.1, verbose=1)
    y_pred = model.predict([X_word,
                            np.array(X_char).reshape((len(X_char),
                                                      max_len, max_len_char))])

    idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels = pred2label(y_pred, idx2tag)
    true_labels = original(y, idx2tag)

    f1_train = f1_score(true_labels, pred_labels)
    precision_train = precision_score(true_labels, pred_labels)
    recall_train = recall_score(true_labels, pred_labels)
    train_scores = [f1_train,precision_train,recall_train]
    print('Training :       ')
    print("F1-score: {:.1%}".format(f1_score(true_labels, pred_labels)))
    print('Precision-score: {:.1%}'.format(precision_score(true_labels, pred_labels)))
    print('Recall-score: {:.1%}'.format(recall_score(true_labels, pred_labels)))

    X_word1 = [[word2idx[w[0]] for w in s] for s in cc_test]
    X_word1 = pad_sequences(maxlen=max_len, sequences=X_word1, value=word2idx["PAD"], padding='post', truncating='post')
    X_char1 = x_char(cc_test, max_len, max_len_char, char2idx)

    y2 = [[tag2idx[w[1]] for w in s] for s in cc_test]
    y2 = pad_sequences(maxlen=max_len, sequences=y2, value=tag2idx["PAD"], padding='post', truncating='post')
    y_pred1 = model.predict([X_word1,
                             np.array(X_char1).reshape((len(X_char1),
                                                        max_len, max_len_char))])
    idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels1 = pred2label(y_pred1, idx2tag)
    true_labels1 = original(y2, idx2tag)

    f1_test = f1_score(true_labels1, pred_labels1)
    precision_test = precision_score(true_labels1, pred_labels1)
    recall_test = recall_score(true_labels1, pred_labels1)
    test_scores = [f1_test,precision_test,recall_test]
    print('Testing :       ')
    print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
    print('Precision-score: {:.1%}'.format(precision_score(true_labels1, pred_labels1)))
    print('Recall-score: {:.1%}'.format(recall_score(true_labels1, pred_labels1)))

    return train_scores, test_scores


def single_d_scores(train_d, test_d):
    if train_d == 'Overall':
        overall_idx = random.sample(set(range(1, 111)), 20)
        train_loc = generate_datasets(dir, 'Overall', 110, overall_idx)
    else:
        train_idx = random.sample(set(range(1, 11)), 2)
        train_loc = generate_datasets(dir, train_d, 110, train_idx)
    test_idx = random.sample(set(range(1, 11)), 2)
    test_loc = generate_datasets(dir, test_d, 10, test_idx)
    print('===========Trained on {} Tested on {}'.format(train_d,test_d))
    train_scores, test_scores = bilstm_character(train_loc, test_loc)
    return train_scores, test_scores


def scores_collect(domains,not_all,):
    f1_tr, f1_te = [], []
    precision_tr, precision_te = [], []
    recall_tr, recall_te = [], []
    if not_all:
        domains = set(domains) - set('Overall')
    for i in range(len(domains)):
        for j in range(len(domains)):
            f1_te.append(single_d_scores(domains[i],domains[j])[1][0])


def domain_specific(dir,domains):
    f1_tr, f1_te = [], []
    precision_tr, precision_te = [], []
    recall_tr, recall_te = [], []
    for d in domains:
        if d == 'Overall':
            num_of_files = 110
        else:
            num_of_files = 10
        test_idx = random.sample(set(range(1, num_of_files + 1)), int(num_of_files * 0.2))
        test_file, train_file = generate_datasets(dir, d, num_of_files, test_idx)
        print('===========', d)
        train_scores, test_scores = bilstm_character(train_file, test_file)
        f1_tr.append(train_scores[0])
        f1_te.append(test_scores[0])
        precision_tr.append(train_scores[1])
        precision_te.append(test_scores[1])
        recall_tr.append(train_scores[2])
        recall_te.append(test_scores[2])
    plot_scores(f1_tr, precision_tr, recall_tr, domains, 'BiLSTM-ChE-Domain-Specific-Scores(train)')
    plot_scores(f1_te, precision_te, recall_te, domains, 'BiLSTM-ChE-Domain-Specific-Scores(test)')
    f1_1 = [round(s, 2) for s in f1_te]
    recall = [round(s, 2) for s in recall_te]
    precision = [round(s, 2) for s in precision_te]
    f1_1 = ['%.2f' % i for i in f1_tr]
    f1_2 = ['%.2f' % i for i in f1_te]
    recall_1 = ['%.2f' % i for i in recall_tr]
    recall_2 = ['%.2f' % i for i in recall_te]
    precision_1 = ['%.2f' % i for i in precision_tr]
    precision_2 = ['%.2f' % i for i in precision_te]
    print('Domain-specific:', '\n', 'Training Validation Scores:', '\n', f1_1, '\n', recall_1, '\n', precision_1)
    print('Testing Scores:', '\n', f1_2, '\n', precision_2, '\n', recall_2)


def domain_independent(dir,domains):
    # domain-independent classifier: train with overall datasets, and test with specific domain
    f1_tr, f1_te = [], []
    precision_tr, precision_te = [], []
    recall_tr, recall_te = [], []
    overall_idx = random.sample(set(range(1, 111)), 20)
    test_overall, train_overall = generate_datasets(dir, 'Overall', 110, overall_idx)
    for i in range(len(domains) - 1):
        test_d, train_d = generate_datasets(dir, domains[i], 11, random.sample(set(range(1, 11)), 2))
        print('===========', domains[i])
        train_scores, test_scores = bilstm_character(train_overall, test_d)
        f1_tr.append(train_scores[0])
        f1_te.append(test_scores[0])
        precision_tr.append(train_scores[1])
        precision_te.append(test_scores[1])
        recall_tr.append(train_scores[2])
        recall_te.append(test_scores[2])
    s_domains = set(domains) - set('Overall')
    plot_scores(f1_tr, precision_tr, recall_tr, s_domains, 'BiLSTM-ChE-domain-Independent-Scores(train)')
    plot_scores(f1_te, precision_te, recall_te, s_domains, 'BiLSTM-ChE-domain-Independent-Scores(test)')
    f1_1 = ['%.2f' % i for i in f1_tr]
    f1_2 = ['%.2f' % i for i in f1_tr]
    recall_1 = ['%.2f' % i for i in recall_tr]
    recall_2 = ['%.2f' % i for i in recall_te]
    precision_1 = ['%.2f' % i for i in precision_tr]
    precision_2 = ['%.2f' % i for i in precision_te]
    print('Domain-independent:', '\n', 'Training Validation Scores:', '\n', f1_1, '\n', recall_1, '\n', precision_1)
    print('Testing Scores:', '\n', f1_2, '\n', precision_2, '\n', recall_2)


def f1_domains(dir,domains):
    # train with one domain and test on other domains
    f1_te = {'Math': [], 'Med': [], 'ES': [], 'Chem': [], 'CS': [], 'Astr': [], 'Agr': [], 'MS': [], 'Bio': [],
             'Eng': []}  # test domain as the key
    for i in range(len(domains) - 1):
        test_d = generate_datasets(dir, domains[i], 10, random.sample(set(range(1, 11)), 2))[0]
        for j in range(len(domains) - 1):
            train_d = generate_datasets(dir, domains[i], 10, random.sample(set(range(1, 11)), 2))[1]
            print('============== Trained on {}, Tested on {}'.format(domains[j], domains[i]))
            test_scores = bilstm_character(train_d, test_d)[1]
            f1_te[domains[i]].append(test_scores[0])
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
    ax.set_title('BiLSTM-ChE F1 Scores Trained and Tested on Each Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(f1_te.keys())
    ax.legend(bbox_to_anchor=(1.2, 1))
    plt.savefig('/data/xwang/OA-STM-domains/{}'.format('BiLSTM-ChE-per-domain'))
    plt.show()
    for k in f1_te.keys():
        f1_te[k] = ['%.2f' % i for i in f1_te[k]]
    print('F1 scores:', '\n', f1_te)


if __name__ == '__main__':
    dir = '/data/xwang/OA-STM-domains/'
    domains = ['Math', 'Med', 'ES', 'Chem', 'CS', 'Astr', 'Agr', 'MS', 'Bio', 'Eng', 'Overall']
    #domain_specific(dir,domains)
    #domain_independent(dir,domains)
    f1_domains(dir,domains)
    print('Done')



