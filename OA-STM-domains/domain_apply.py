from CRF_kfold import plot_scores, generate_datasets
import random
from BiLSTM_CRF import bilstm_crf


def domain_specific(dir, domains, fun, plotname):
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
        train_scores, test_scores = fun(train_file, test_file)
        f1_tr.append(train_scores[0])
        f1_te.append(test_scores[0])
        precision_tr.append(train_scores[1])
        precision_te.append(test_scores[1])
        recall_tr.append(train_scores[2])
        recall_te.append(test_scores[2])
    plot_scores(f1_tr, precision_tr, recall_tr, domains, plotname+'(train)')
    plot_scores(f1_te, precision_te, recall_te, domains, plotname+'(test)')
    f1_1 = ['%.2f' % i for i in f1_tr]
    f1_2 = ['%.2f' % i for i in f1_tr]
    recall_1 = ['%.2f' % i for i in recall_tr]
    recall_2 = ['%.2f' % i for i in recall_te]
    precision_1 = ['%.2f' % i for i in precision_tr]
    precision_2 = ['%.2f' % i for i in precision_te]
    print('Domain-specific:', '\n', 'Training Validation Scores:', '\n', f1_1, '\n', recall_1, '\n', precision_1)
    print('Testing Scores:', '\n', f1_2, '\n', precision_2, '\n', recall_2)

