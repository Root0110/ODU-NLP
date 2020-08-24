#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 14:08
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : BiLSTM-Elmo-Attention.py
# @Software: PyCharm

from file_prepare import preprocess, cuu, combine_all, pred2label
from CRF_kfold import plot_scores, generate_datasets

import numpy as np
import random
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
import keras
from keras_self_attention import SeqSelfAttention
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


def original(y,idx2tag):
    sent_tag_list = []
    for sent_tags in y:
        word_tag_list = []
        for tag in sent_tags:
            word_tag_list.append(idx2tag[tag])
        sent_tag_list.append(word_tag_list)
    return sent_tag_list


def bilstm_attention(train_loc, test_loc):
    train_pre = preprocess(train_loc)
    test_pre = preprocess(test_loc)
    cc_train = cuu(train_pre)
    cc_test = cuu(test_pre)
    words_all, tags_all = combine_all(cc_train, cc_test)
    n_tags = len(tags_all)
    n_words = len(words_all)

    max_len = 130
    tag2idx = {t: i for i, t in enumerate(tags_all)}
    X = [[w[0] for w in s] for s in cc_train]
    new_X = []
    for seq in X:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_X.append(new_seq)
    X = new_X
    y = [[tag2idx[w[1]] for w in s] for s in cc_train]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    batch_size = 32

    sess = tf.Session()
    K.set_session(sess)
    elmo_model = hub.Module("/data/xwang/module_elmo2", trainable=True)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    def ElmoEmbedding(x):
        return elmo_model(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(batch_size * [max_len])
        },
            signature="tokens",
            as_dict=True)["elmo"]
    input_text = Input(shape=(max_len,), dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    x = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(embedding)
    x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l1(1e-4),
                           attention_regularizer_weight=1e-4,
                           name='Attention')(x)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(att)
    model = Model(input_text, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    train_num = (len(X) // batch_size) * batch_size
    print(n_words, n_tags, len(X), len(y), train_num)

    X_tr = X[:train_num]
    y_tr = y[:train_num]
    y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
    history = model.fit(np.array(X_tr), y_tr,
                        batch_size=batch_size, epochs=7, verbose=1)
    test_pred = model.predict(np.array(X_tr), verbose=1)
    idx2tag = {i: w for w, i in tag2idx.items()}
    X2 = [[w[0] for w in s] for s in cc_test]
    new_X = []
    for seq in X2:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_X.append(new_seq)
    test_num = (len(new_X) // batch_size) * batch_size
    print(len(X2), len(new_X), test_num)
    X2 = new_X[:test_num]
    y2 = [[tag2idx[w[1]] for w in s] for s in cc_test]
    y2 = pad_sequences(maxlen=max_len, sequences=y2, padding="post", value=tag2idx["O"])
    y2 = y2[:test_num]
    test_pred1 = model.predict(np.array(X2), verbose=1)
    pred_labels1 = pred2label(test_pred1, idx2tag)
    true_labels1 = original(y2,idx2tag)
    f1_test = f1_score(true_labels1, pred_labels1)
    precision_test = precision_score(true_labels1, pred_labels1)
    recall_test = recall_score(true_labels1, pred_labels1)
    test_scores = [f1_test, recall_test, precision_test]
    print('Testing:', test_scores)
    return test_scores


def kfold_bilstm_att(dir,d_train,d_test,num_tr,num_te):
    # given addresses of all the files, perform k-fold validation, return average scores
    f1 = []
    recall = []
    precision = []
    idx1 = set(range(1,num_tr+1))
    idx2 = set(range(1,num_te+1))
    test_num1 = int(num_tr*0.2)
    test_num2 = int(num_te*0.2)
    for i in range(5):
        temp1 = random.sample(idx1, test_num1) # the first 0.2 for testing
        temp2 = random.sample(idx2, test_num2)
        idx1 = idx1 - set(temp1) # the remaining 0.8 indexes for training
        idx2 = idx2 - set(temp2)
        train_loc = generate_datasets(dir, d_train, num_tr,temp1)[1]
        test_loc = generate_datasets(dir, d_test, num_te,temp2)[0]
        s = bilstm_attention(train_loc, test_loc)
        f1.append(s[0])
        recall.append(s[1])
        precision.append(s[2])
    print('kfold mean scores:', np.mean(f1), np.mean(recall), np.mean(precision))
    return np.mean(f1), np.mean(recall), np.mean(precision)


def domain_specific(dir,domains):
    f1_te = []
    precision_te = []
    recall_te = []
    for d in domains:
        if d == 'Overall':
            num_of_files = 110
        else:
            num_of_files = 10
        train_loc = dir + d + '/train_com.txt'
        test_loc = dir + d + '/test_com.txt'
        scores = bilstm_attention(train_loc, test_loc)
        print('======== {}'.format(d))
        f1_te.append(scores[0])
        recall_te.append(scores[1])
        precision_te.append(scores[2])
    plot_scores(f1_te, recall_te, precision_te, domains, 'BilSTM-Elmo-Attention-Performance')
    f1 = [round(s, 2) for s in f1_te]
    recall = [round(s, 2) for s in recall_te]
    precision = [round(s, 2) for s in precision_te]
    print('Domain-specific:', '\n', f1, '\n', recall, '\n', precision)



if __name__ == '__main__':
    dir = '/data/xwang/OA-STM-domains/'
    #domains = ['Math', 'Med', 'ES', 'Chem', 'CS', 'Astr', 'Agr', 'MS', 'Bio', 'Eng']
    domains = ['Chem']
    domain_specific(dir,domains)

