#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 18:22
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm

import numpy as np


def preprocess(file_loc):
    # store each group of "word-tag" into a tuple
    words = []
    wordtags = []
    nanjo = []
    x = open(file_loc, encoding="utf8")
    for i in x:
        # i=re.sub(r'[^\w\s]','',i)
        a = i.split('\t')
        # print(a)
        w = a[0].rstrip()
        words.append(w)
        if len(a) <= 1:  # deal with the '\n' in the first row
            continue
        else:
            wt = a[1].rstrip()
        wordtags.append(wt)
        nanjo.append((w, wt))
    return nanjo


def cuu(nanjo):
    #store word-tags in the same one sentence into one list [[(),()]]
    cc = []
    cc1 = []
    for i in nanjo:
        if i != ('',''):
            cc1.append(i)
            #print(cc1)
        else:
            cc.append(cc1)
            cc1=[]
    ss=('.','O')
    for i in range(len(cc)):
        cc[i].append(ss)
    for i in range(len(cc)):
        for j in range(len(cc[i])):
            if cc[i][j] == ('.', 'O'):
                w=cc[i][j-1][0].replace(".","")
                t = cc[i][j-1][1]
                cc[i][j-1]=(w,t)
    return cc


def combine_all(cc_train, cc_test):
    words_all = []
    tags_all = []
    for i in range(len(cc_train)):
        for j in range(len(cc_train[i])):
            words_all.append(cc_train[i][j][0])
            tags_all.append(cc_train[i][j][1])

    for i in range(len(cc_test)):
        for j in range(len(cc_test[i])):
            words_all.append(cc_test[i][j][0])
            tags_all.append(cc_test[i][j][1])

    words_all = list(set((words_all)))
    words_all.append("ENDPAD")  # add ending mark
    tags_all = list(set((tags_all)))
    return words_all,tags_all


def pred2label(pred,idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out







