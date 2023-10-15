import os
import time

from corpus_utils import read_corpus, move_empty_files
from nlp_utils import preprocessing, preprocessing_v2, no_spacing
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from src.evaluation_utils import compute_evaluation_measures, compute_means_std_eval_measures


if __name__ == '__main__':

    corpus_path = '/media/hilario/Novo Volume/Hilario/Pesquisa/Experimentos/renato/resumes_corpus'

    vectorizer_options = ['tf_idf']

    n_splits = 5

    n_total = -1

    max_features = 5000

    print('\nLoading Corpus\n')

    corpus_df = read_corpus(corpus_path, num_examples=n_total)

    print('\nPreProcessing Corpus\n')

    corpus_df['resume_norm'] = corpus_df['resume'].apply(lambda t: preprocessing_v2(t)).astype(str)
    corpus_df['label_unique'] = corpus_df['label'].apply(lambda l: l[0]).astype(str)
    corpus_df['no_spacing'] = corpus_df['resume_norm'].apply(lambda t: no_spacing(t)).astype(str)

    corpus_df_unique = corpus_df.drop_duplicates(subset='no_spacing')

    corpus_df_unique['resume_nlp'] = corpus_df_unique['resume'].apply(
        lambda t: preprocessing(t)).astype(str)

    resumes = corpus_df_unique['resume_nlp'].values
    labels = corpus_df_unique['label_unique'].values

    print(f'\nCorpus: {len(resumes)} -- {len(labels)}')

    print('\nExample:')
    print(f'  Resume: {resumes[-1]}')
    print(f'  Label: {labels[-1]}')

    counter_labels = Counter(labels)

    labels_distribution = OrderedDict(sorted(counter_labels.items()))

    print(f'\nLabels distribution: {labels_distribution}')

    for vectorizer_opt in vectorizer_options:

        print(f'\n\nVectorizer: {vectorizer_opt}\n\n')

        results_dir = f'../results/ml/{vectorizer_opt}'

        os.makedirs(results_dir, exist_ok=True)

        vectorizer = None

        if vectorizer_opt == 'tf_idf':
            vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features)
        elif vectorizer_opt == 'count':
            vectorizer = CountVectorizer(ngram_range=(1, 1), binary=False, max_features=max_features)
        else:
            vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, max_features=max_features)

        label_encoder = LabelEncoder()

        print(f'\nVectorizer Option: {vectorizer_opt}')

        y_labels = label_encoder.fit_transform(labels)

        print(f'\nLabels Mappings: {label_encoder.classes_}')

        classifiers = {
            'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=500),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(class_weight='balanced'),
            'random_forest': RandomForestClassifier(class_weight='balanced'),
            'extra_trees_classifier': ExtraTreesClassifier(class_weight='balanced'),
            'xgboost': XGBClassifier(),
            'lgbm': LGBMClassifier(class_weight='balanced'),
            'svc': SVC(class_weight='balanced'),
            'cat_boost_classifier': CatBoostClassifier(verbose=False),
            'mlp_classifier': MLPClassifier()
        }

        print('\n\n------------Evaluations------------\n')

        for clf_name, clf_base in classifiers.items():

            results_dict = {
                'all_accuracy': [],
                'all_macro_avg_p': [],
                'all_macro_avg_r': [],
                'all_macro_avg_f1': [],
                'all_weighted_avg_p': [],
                'all_weighted_avg_r': [],
                'all_weighted_avg_f1': []
            }

            print(f'\n  Classifier: {clf_name}')

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            all_y_test = []
            all_y_pred = []

            for k, (train_idx, test_idx) in enumerate(skf.split(resumes, y_labels)):

                classifier = clone(clf_base)

                X_train = [resume for i, resume in enumerate(resumes) if i in train_idx]
                X_test = [resume for i, resume in enumerate(resumes) if i in test_idx]

                y_train = y_labels[train_idx]
                y_test = y_labels[test_idx]

                X_train = vectorizer.fit_transform(X_train).toarray()
                X_test = vectorizer.transform(X_test).toarray()

                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, test_size=0.1, stratify=y_train, shuffle=True,
                    random_state=42)

                print(f'\n    Folder {k + 1} - {len(X_train)} - {len(X_test)}')

                classifier.fit(X_train, y_train)

                y_pred = classifier.predict(X_test)

                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred)

                compute_evaluation_measures(y_test, y_pred, results_dict)

            compute_means_std_eval_measures(clf_name, all_y_test, all_y_pred, results_dict, results_dir)

        time.sleep(60)
