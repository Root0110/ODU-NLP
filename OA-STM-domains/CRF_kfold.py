#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 16:39
# @Author  : Xiaohan Wang
# @Site    : 
# @File    : k-fold cv.py
# @Software: PyCharm

import random
import numpy as np
from CRFNER import crf
import matplotlib.pyplot as plt


def generate_datasets(dir, domain, num_of_files, test_idx):
    # combine 80% files into the whole training dataset during one-fold-cross-validation, and the other 20% into testing one
    test = []
    train = []
    for i in range(1, num_of_files):
        file_loc = dir + domain + '/file{}.txt'.format(i)
        if i in test_idx:
            with open(file_loc, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if 'DOCSTART' not in l:
                        test.append(l)
        else:
            with open(file_loc, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if 'DOCSTART' not in l:
                        train.append(l)
    with open(dir + domain + '/test_com.txt', 'w') as f1:
        f1.writelines(test)
    with open(dir + domain + '/train_com.txt', 'w') as f2:
        f2.writelines(train)
    combined_loc = ['/data/xwang/OA-STM-domains/' + domain + '/test_com.txt',
                    '/data/xwang/OA-STM-domains/' + domain + '/train_com.txt']
    return combined_loc


def plot_scores(f1, recall, precision, labels, plot_name):
    width = 1/4
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x - 3*width/2, f1, width, label='f1')
    ax.bar(x - width/2, recall, width, label='recall')
    ax.bar(x + width/2, precision, width, label='precision')
    ax.set_ylabel('Scores')
    ax.set_title('{}'.format(plot_name))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center')
    # add value at the top of the bar
    '''rects = ax.patches
    total = f1 + recall + precision
    values = ['%.2f' % i for i in total ]
    for rect, value in zip(rects, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, value,
                ha='center', va='bottom')'''
    plt.savefig('/data/xwang/OA-STM-domains/{}'.format(plot_name))
    plt.show()


def kfold_crf(dir,d_train,d_test,num_tr,num_te):
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
        s = crf(test_loc, train_loc)
        f1.append(s[0])
        recall.append(s[1])
        precision.append(s[2])
    return np.mean(f1), np.mean(recall), np.mean(precision)


def domain_specific(dir, domains):
    f1_total = []
    recall_total = []
    precision_total = []
    for d in domains:
        if d == 'Overall':
            num_of_files = 110
        else:
            num_of_files = 10
        scores = kfold_crf(dir, d, d, num_of_files, num_of_files)
        print('======== Trained on {}, Tested on {}'.format(d,d))
        #print('f1 score:', scores[0], 'recall:', scores[1], 'precision:', scores[2])
        f1_total.append(scores[0])
        recall_total.append(scores[1])
        precision_total.append(scores[2])
    plot_scores(f1_total, recall_total, precision_total, domains, 'CRF-Domain-Specific-Scores')
    # output scores with 2 decimal points
    f1 = [round(s, 2) for s in f1_total]
    recall = [round(s, 2) for s in recall_total]
    precision = [round(s, 2) for s in precision_total]
    print('Domain-specific:', '\n', f1, '\n', recall, '\n', precision)


def domain_independent(dir, domains):
    f1_total = []
    recall_total = []
    precision_total = []
    all_train_loc = '/data/xwang/OA-STM-domains/Overall/train_com.txt'
    d_train = 'Overall'
    for d in domains:
        if d != 'Overall':
            scores = kfold_crf(dir,d_train,d,110,10)
            print('======== Trained on Overall, Tested on {}'.format(d))
            f1_total.append(scores[0])
            recall_total.append(scores[1])
            precision_total.append(scores[2])
    plot_scores(f1_total, recall_total, precision_total, domains[:-1], 'CRF-Domain-Independent-Scores')
    f1 = [round(s,2) for s in f1_total]
    recall = [round(s,2) for s in recall_total]
    precision = [round(s,2) for s in precision_total]
    print('Domain-independent:','\n', f1,'\n',recall,'\n',precision)


def f1_domains(dir,domains):
    f1_te = {'Math': [], 'Med': [], 'ES': [], 'Chem': [], 'CS': [], 'Astr': [], 'Agr': [], 'MS': [], 'Bio': [],
             'Eng': []}  # test domain as the key
    for i in range(len(domains) - 1):
        #test_d = generate_datasets(dir, domains[i], 10, random.sample(set(range(1, 11)), 2))[0]
        for j in range(len(domains) - 1):
            #train_d = generate_datasets(dir, domains[i], 10, random.sample(set(range(1, 11)), 2))[1]
            print('============== Trained on {}, Tested on {}'.format(domains[j], domains[i]))
            scores = kfold_crf(dir,domains[j],domains[i],10,10)
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
    plt.savefig('/data/xwang/OA-STM-domains/{}'.format('CRF-per-domain'))
    plt.show()
    for k in f1_te.keys():
        f1_te[k] = [round(s,2) for s in f1_te[k]]
    print('F1 scores:', '\n', f1_te)


if __name__ == '__main__':
    dir = '/data/xwang/OA-STM-domains/'
    domains = ['Math', 'Med', 'ES', 'Chem', 'CS', 'Astr', 'Agr', 'MS', 'Bio', 'Eng', 'Overall']
    #domain-specific
    domain_specific(dir,domains)
    # domain-independent
    domain_independent(dir,domains)
    # domain model to other domains
    f1_domains(dir,domains)
    print('Done')