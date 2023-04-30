import os
import json
import matplotlib.pyplot as plt
import numpy as np
from corpus_utils import read_corpus
from nlp_utils import preprocessing
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from neuralnetworks import MLP, BidirectionalLSTM

def evaluate_model(X_resumes, y_labels, num_classes, model, n_splits=5, callbacks=[]):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_test = []
    all_y_pred = []
    results = {
        'all_accuracy': [],
        'all_macro_precision': [],
        'all_macro_recall': [],
        'all_macro_f1': []
    }
    for train_index, test_index in skf.split(X_resumes, y_labels):
        X_train, X_test = X_resumes[train_index], X_resumes[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        model.compileModel()
        model.fitModel(X_train, y_train, 20)
        y_pred = model.predictModel(X_test)
        y_test_bool = np.argmax(y_test, axis=1)
        y_pred_bool = np.argmax(y_pred, axis=1)
        all_y_test.extend(y_test_bool)
        all_y_pred.extend(y_pred_bool)
        results['all_accuracy'].append(model.evaluateModel(X_test, y_test)[1])
    results['all_macro_precision'], results['all_macro_recall'], results['all_macro_f1'], _ = classification_report(all_y_test, all_y_pred, output_dict=True)['macro avg'].values()
    return results

def callClassifier(clf_name, clf_base, cb: bool = False) -> None:        
    print(f"\n\n{clf_name}\n")
    model = clf_base
    early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=1, mode='min', baseline=None)
    if cb:
        callbacks = [early_stopping]
        results = evaluate_model(X_resumes, y_labels, num_classes, model, n_splits=5, callbacks=callbacks)
    else:
        results = evaluate_model(X_resumes, y_labels, num_classes, model, n_splits=5)
    print(results)

if __name__ == '__main__':
    print("=============================================== vai planetaaaaaa ===============================================")
    # vectorizer_opt = 'binary'
    vectorizer_opt = 'count'
    # vectorizer_opt = 'tf_idf'
    corpus_path = 'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus'
    results_dir = f'E:\\Renato\\Mestrado\\dissertacao_v2\\data\\results\\neuralnetworks\\{vectorizer_opt}'
    os.makedirs(results_dir, exist_ok=True)
    n_splits = 5
    n_total = 600
    max_features = None
    print('\nLoading Corpus\n')
    corpus_df = read_corpus(corpus_path, num_examples=n_total)
    corpus_df['resume_nlp'] = corpus_df['resume'].apply(lambda t: preprocessing(t)).astype(str)
    corpus_df['label_unique'] = corpus_df['label'].apply(lambda l: l[0]).astype(str)
    resumes = corpus_df['resume_nlp'].values
    labels = corpus_df['label_unique'].values
    if n_total > 0:
        resumes = resumes[:n_total]
        labels = labels[:n_total]
    print(f'\nCorpus: {len(resumes)} -- {len(labels)}')
    print('\nExample:')
    print(f'  Resume: {resumes[-1]}')
    print(f'  Label: {labels[-1]}')
    counter_1 = Counter(labels)
    labels_distribution = OrderedDict(sorted(counter_1.items()))
    print(f'\nLabels distribution: {labels_distribution}')
    vectorizer = None
    if vectorizer_opt == 'tf_idf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features)
    elif vectorizer_opt == 'count':
        vectorizer = CountVectorizer(ngram_range=(1, 1), binary=False, max_features=max_features)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, max_features=max_features)
    label_encoder = LabelEncoder()
    print(f'\nVectorizer Option: {vectorizer_opt}')
    X_resumes = vectorizer.fit_transform(resumes).toarray()
    print(f'X Shape: {X_resumes.shape}')
    y_labels = label_encoder.fit_transform(labels)
    print(f'Y Shape: {y_labels.shape}')
    y_true = label_encoder.inverse_transform(y_labels)
    print('\nExample Encoded:')
    print(f'  Resume: {X_resumes[-1]}')
    print(f'  Label: {y_labels[-1]}')
     
    num_classes = len(np.unique(y_true))
    print(f'num classes: {num_classes}')
    classifiers = {
        # 'Multilayer Perceptron': MLP(num_classes),
        'Bidirectional LSTM': BidirectionalLSTM(num_classes)
    }
    print('\n\n------------Evaluations------------\n')
    for clf_name, clf_base in classifiers.items():
        callClassifier(clf_name, clf_base)