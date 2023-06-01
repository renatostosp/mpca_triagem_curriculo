import os
import numpy as np

from corpus_utils import read_corpus
from nlp_utils import preprocessing_v2
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.keras_models_helper import build_feed_foward, build_feed_foward_emb, build_cnn_model, build_lstm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from src.evaluation_utils import compute_evaluation_measures, compute_means_std_eval_measures


if __name__ == '__main__':

    corpus_path = '../resumes_corpus'

    n_total = 200

    n_splits = 5

    # model_name = 'feed_foward'
    # model_name = 'feed_foward_emb'
    # model_name = 'cnn'
    model_name = 'lsm'

    results_dir = f'../results/nn/{model_name}'

    num_epochs = 1

    batch_size = 64

    vocab_size = 1000
    emb_dim = 100

    checkpoint_dir = '../checkpoints/'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print('\nLoading Corpus\n')

    corpus_df = read_corpus(corpus_path, num_examples=n_total)

    print('\nPreProcessing Corpus\n')

    corpus_df['resume_nlp'] = corpus_df['resume'].apply(lambda t: preprocessing_v2(t)).astype(str)
    corpus_df['label_unique'] = corpus_df['label'].apply(lambda l: l[0]).astype(str)

    resumes = corpus_df['resume_nlp'].values
    labels = corpus_df['label_unique'].values

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

        model = None

        if model_name == 'feed_foward':
            model = build_feed_foward(max_len, num_classes)
        elif model_name == 'feed_foward_emb':
            model = build_feed_foward_emb(vocab_size, max_len, num_classes, emb_dim)
        elif model_name == 'cnn':
            model = build_cnn_model(vocab_size, max_len, num_classes, emb_dim, num_filters=16,
                                    kernel_size=3)
        elif model_name == 'lsm':
            model = build_lstm(vocab_size, max_len, num_classes, emb_dim)

        model_checkpoint = ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, monitor='val_accuracy',
                                           mode='max',save_best_only=True)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val),
                            callbacks=[model_checkpoint])

        model.load_weights(checkpoint_dir)

        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)

        y_pred = [y for y in y_pred]

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        compute_evaluation_measures(y_test, y_pred, results_dict)

        keras.backend.clear_session()

    compute_means_std_eval_measures(model_name, all_y_test, all_y_pred, results_dict, results_dir)