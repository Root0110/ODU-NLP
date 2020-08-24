#from itertools import chain

#import nltk
#import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import scipy
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from pprint import pprint
import random
import pickle
import nltk
nltk.download('averaged_perceptron_tagger')


from FeatureExtraction import sent2labels,sent2features
#from PhraseEval import phrasesFromTestSenJustExtraction,phrase_extraction_report
from DataExtraction import convertCONLLFormJustExtractionSemEval

def crf(test_loc,train_loc):
    test_sents = convertCONLLFormJustExtractionSemEval(test_loc)
    train_sents = convertCONLLFormJustExtractionSemEval(train_loc)
    
    #pprint(train_sents[0])
    #pprint(test_sents[0])
        
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    crf = sklearn_crfsuite.CRF(\
    algorithm='lbfgs',\
    c1=0.1,\
    c2=0.1,\
    max_iterations=100,\
    all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    labels = list(crf.classes_)
    labels.remove('O')
    #print(labels)
    pickle.dump(crf,open("/data/xwang/models_origin/linear-chain-crf.model.pickle","wb"), protocol = 0, fix_imports = True)
    y_pred = crf.predict(X_test)

    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
    f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=sorted_labels)
    recall = metrics.flat_recall_score(y_test, y_pred, average='weighted',labels=sorted_labels)
    precision = metrics.flat_precision_score(y_test, y_pred, average='weighted',labels=sorted_labels)
    #print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    return (f1_score, recall, precision)

if __name__ == "__main__":
    crf('/data/xwang/OA-STM-domains/Arg/test_com.txt','/data/xwang/OA-STM-domains/Arg/train_com.txt')

