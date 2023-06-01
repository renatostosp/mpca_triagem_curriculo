import os
import numpy as np
import pandas as pd

from corpus_utils import read_corpus, move_empty_files
from nlp_utils import preprocessing_v2, no_spacing
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.utils import to_categorical, pad_sequences
from keras.preprocessing.text import Tokenizer
from neuralnetworks import Checkpoint, FeedForward, EmbFeedForward, BidirectionalLSTM, CNN
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from evaluation_utils import compute_evaluation_measures, compute_means_std_eval_measures
from keras.models import Sequential
import tensorflow as tf


if __name__ == '__main__':

    corpus_path = 'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus'
    empty_path = 'E:\\Renato\\Mestrado\\dissertacao_v2\\empty_files'

    n_total = 600

    n_splits = 5

    model_name = 'feed_forward'
    # model_name = 'feed_forward_emb'
    # model_name = 'cnn'
    # model_name = 'lstm'

    results_dir = f'E:\\Renato\\Mestrado\\dissertacao_v2\\data\\results\\neuralnetworks\\{model_name}'

    num_epochs = 5

    batch_size = 64

    vocab_size = 1000
    emb_dim = 100

    checkpoint_dir = f'E:\\Renato\\Mestrado\\dissertacao_v2\\mpca_triagem_curriculo\\checkpoints\\{model_name}'
    checkpoint_path = checkpoint_dir + '\\training_1\\cp.ckpt'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print('\nRemoving empty files\n')

    move_empty_files(corpus_path, empty_path)     

    print('\nLoading Corpus\n')

    corpus_df = read_corpus(corpus_path, num_examples=n_total)

    print('\nPreProcessing Corpus\n')

    corpus_df['resume_nlp'] = corpus_df['resume'].apply(lambda t: preprocessing_v2(t)).astype(str)
    corpus_df['label_unique'] = corpus_df['label'].apply(lambda l: l[0]).astype(str)
    corpus_df['no_spacing'] = corpus_df['resume_nlp'].apply(lambda t: no_spacing(t)).astype(str)
    corpus_df_unique = corpus_df.drop_duplicates(subset='no_spacing')

    resumes = corpus_df_unique['resume_nlp'].values
    labels = corpus_df_unique['label_unique'].values

    num_classes = len(set(labels))

    print(f'\nCorpus: {len(resumes)} -- {len(labels)} -- {num_classes}')

    print('\nExample:')
    print(f'  Resume: {resumes[-1]}')
    print(f'  Label: {labels[-1]}')

    counter_labels = Counter(labels)

    labels_distribution = OrderedDict(sorted(counter_labels.items()))

    print(f'\nLabels Distribution: {labels_distribution}')

    label_encoder = LabelEncoder()

    y_labels = label_encoder.fit_transform(labels)

    print(f'\nLabels Mapping: {label_encoder.classes_}')

    print(f'\nModel Name: {model_name}')

    print('\n\n------------Evaluations------------\n')

    results_dict = {
        'all_accuracy': [],
        'all_macro_avg_p': [],
        'all_macro_avg_r': [],
        'all_macro_avg_f1': [],
        'all_weighted_avg_p': [],
        'all_weighted_avg_r': [],
        'all_weighted_avg_f1': []
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_test = []
    all_y_pred = []

    for k, (train_idx, test_idx) in enumerate(skf.split(resumes, y_labels)):

        X_train = [resume for i, resume in enumerate(resumes) if i in train_idx]
        X_test = [resume for i, resume in enumerate(resumes) if i in test_idx]

        y_train = y_labels[train_idx]
        y_test = y_labels[test_idx]

        X_train, X_val, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, shuffle=True, random_state=42)

        print(f'\n  Folder {k + 1} - {len(X_train)} - {len(X_val)} - {len(X_test)}')

        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_valid, num_classes=num_classes)

        tokenizer = Tokenizer(oov_token='<OOV>')

        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_val = tokenizer.texts_to_sequences(X_val)
        X_test = tokenizer.texts_to_sequences(X_test)

        max_len = max([len(x) for x in X_train])

        X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
        X_val = pad_sequences(X_val, maxlen=max_len, padding='post')
        X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

        nn_model = None

        if model_name == 'feed_forward':
            nn_model = FeedForward(max_len, num_classes)
        elif model_name == 'feed_forward_emb':
            nn_model = EmbFeedForward(max_len, vocab_size, emb_dim, num_classes)
        elif model_name == 'cnn':
            nn_model = CNN(vocab_size, max_len, num_classes, emb_dim, num_filters=16,
                                    kernel_size=3)
        elif model_name == 'lstm':
            nn_model = BidirectionalLSTM(vocab_size, max_len, num_classes, emb_dim)

        callback = None
        # callback = ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, monitor='val_accuracy',
        #                                    mode='max',save_best_only=True)

        history = nn_model.fit_model(X_train, y_train, X_val, y_val, num_epochs, batch_size, checkpoint_dir=checkpoint_dir)
                        
        if not nn_model.cb or isinstance(callback, ModelCheckpoint):
            nn_model.model.load_weights(checkpoint_dir)

        y_pred = nn_model.predict_model(X_test)

        y_pred = np.argmax(y_pred, axis=1)

        y_pred = [y for y in y_pred]

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        compute_evaluation_measures(y_test, y_pred, results_dict)

        keras.backend.clear_session()

    compute_means_std_eval_measures(model_name, all_y_test, all_y_pred, results_dict, results_dir)
