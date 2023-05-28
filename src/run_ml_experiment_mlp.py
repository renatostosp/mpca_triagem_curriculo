import os
import json
import matplotlib.pyplot as plt
import numpy as np
from corpus_utils import read_corpus
from nlp_utils import preprocessing
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from keras.utils import to_categorical, pad_sequences
from neuralnetworks import FeedForward, EmbFeedForward, BidirectionalLSTM, CNN
from keras.preprocessing.text import Tokenizer

def evaluate_model(X_resumes, y_labels, num_classes, vocab_size, embedding_dim, input_length, model, n_splits=5, callbacks=[]):

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_test = []
    all_y_pred = []
    results = {
        'all_accuracy': [],
        'all_macro_precision': [],
        'all_macro_recall': [],
        'all_macro_f1': []
    }
    
    for train_index, test_index in skf_outer.split(X_resumes, y_labels):
        X_train, X_test = X_resumes[train_index], X_resumes[test_index]
        y_train, y_test = y_labels[train_index], y_labels[test_index]

        skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=52)
        for train_inner_index, val_index in skf_inner.split(X_train, y_train):
            X_train_split, X_val = X_train[train_inner_index], X_train[val_index]
            y_train_split, y_val = y_train[train_inner_index], y_train[val_index]

            y_train_split = to_categorical(y_train_split, num_classes)
            y_val = to_categorical(y_val, num_classes)
            y_test_categorical = to_categorical(y_test, num_classes)

            model.compileModel()
            model.fitModel(X_train_split, y_train_split, X_val, y_val, epochs=20)
            y_pred = model.predictModel(X_test)

            y_test_bool = np.argmax(y_test_categorical, axis=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            all_y_test.extend(y_test_bool)
            all_y_pred.extend(y_pred_bool)
            results['all_accuracy'].append(model.evaluateModel(X_test, y_test_categorical)[1])

    macro_precision, macro_recall, macro_f1, _ = classification_report(all_y_test, all_y_pred, output_dict=True)['macro avg'].values()
    results['all_macro_precision'].append(macro_precision)
    results['all_macro_recall'].append(macro_recall)
    results['all_macro_f1'].append(macro_f1)

    return results

def callClassifier(clf_name, clf_base) -> None:        
    print(f"\n\n{clf_name}\n")
    model = clf_base
    results = evaluate_model(X_resumes, y_labels, num_classes, vocab_size, embedding_dim, input_length, model, n_splits=5)
    print(results)

if __name__ == '__main__':
    vectorizer_opt = 'neural'
    embedding_dim = 300 # tamanho do vetor de embedding de cada palavra
    input_length = 1000 # tamanho máximo da matriz de embedding toda

    corpus_path = 'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus'
    results_dir = f'E:\\Renato\\Mestrado\\dissertacao_v2\\data\\results\\neuralnetworks\\{vectorizer_opt}'
    os.makedirs(results_dir, exist_ok=True)
    n_splits = 5
    n_total = 50
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
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(labels)
    
    num_classes = len(label_encoder.classes_)
    print(f'num classes: {num_classes}')

    tokenizer = Tokenizer(oov_token='<oov>')
    tokenizer.fit_on_texts(resumes)
    resumes_sequences = tokenizer.texts_to_sequences(resumes)

    max_len = max([len(x) for x in resumes])

    X_resumes = pad_sequences(resumes_sequences, maxlen=max_len, padding='post')

    print(f'X Shape: {X_resumes.shape}')
    print(f'Y Shape: {y_labels.shape}')

    y_true = label_encoder.inverse_transform(y_labels)
    print('\nExample Encoded:')
    print(f'  Resume: {X_resumes[-1]}')
    print(f'  Label: {y_labels[-1]}')

    vocab_size = len(tokenizer.word_index) + 1
    # vocab_size = 1000

    print('\nVocab size:', vocab_size)

    classifiers = {
        'Feed Forward': FeedForward(num_classes=num_classes, max_len=max_len),
        'Feed Forward with Embedding': EmbFeedForward(num_classes=num_classes, max_len=max_len, vocab_size=vocab_size, embedding_dim=embedding_dim),
        'Bidirectional LSTM': BidirectionalLSTM(num_classes=num_classes, max_len=max_len, vocab_size=vocab_size, embedding_dim=embedding_dim),
        'CNN': CNN(num_classes=num_classes, max_len=max_len, vocab_size=vocab_size, embedding_dim=embedding_dim)

    }

    print('\n\n------------Evaluations------------\n')
    for clf_name, clf_base in classifiers.items():
        callClassifier(clf_name, clf_base)
