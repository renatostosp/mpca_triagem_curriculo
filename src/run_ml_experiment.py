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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


if __name__ == '__main__':

    # vectorizer_opt = 'binary'
    # vectorizer_opt = 'count'
    vectorizer_opt = 'tf_idf'

    corpus_path = 'E:\\Renato\\Mestrado\\dissertacao_v2\\resumes_corpus'

    results_dir = f'E:\\Renato\\Mestrado\\dissertacao_v2\\data\\results\\{vectorizer_opt}'

    os.makedirs(results_dir, exist_ok=True)

    n_splits = 5

    n_total = 1000

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
    y_labels = label_encoder.fit_transform(labels)

    y_true = label_encoder.inverse_transform(y_labels)

    # print(labels)
    # print(y_labels)
    # print(y_true)



    print('\nExample Encoded:')
    print(f'  Resume: {X_resumes[-1]}')
    print(f'  Label: {y_labels[-1]}')

    classifiers = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'KNN': KNeighborsClassifier(weights='uniform'),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=1000, class_weight='balanced'),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=1000, class_weight='balanced'),
        'xgboost': XGBClassifier(n_estimators=1000),
        'lgbm': LGBMClassifier(n_estimators=1000, class_weight='balanced'),
        'svc': SVC(class_weight='balanced'),
        # 'CatBoostClassifier': CatBoostClassifier(n_estimators=1000, verbose=False)
    }
   
    print('\n\n------------Evaluations------------\n')

    for clf_name, clf_base in classifiers.items():

        results = {
        
        'all_accuracy': [],
        'all_macro_precision': [],
        'all_macro_recall': [],
        'all_macro_f1': [],
        
        }

        print(f'\n  Classifier: {clf_name}')

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        all_y_test = []
        all_y_pred = []

        for train_index, test_index in skf.split(X_resumes, y_labels):

            classifier = clone(clf_base)

            X_train, X_test = X_resumes[train_index], X_resumes[test_index]

            y_train, y_test = y_labels[train_index], y_labels[test_index]

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)

            report = classification_report(y_test, y_pred, output_dict=True)
            
            results['all_accuracy'].append(report['accuracy'])
            results['all_macro_precision'].append(report['macro avg']['precision'])
            results['all_macro_recall'].append(report['macro avg']['recall'])
            results['all_macro_f1'].append(report['macro avg']['f1-score'])

        results['mean_accuracy'] = np.mean(results['all_accuracy'])
        results['std_accuracy'] = np.std(results['all_accuracy'])

        results['mean_macro_precision'] = np.mean(results['all_macro_precision'])
        results['std_macro_precision'] = np.std(results['all_macro_precision'])

        results['mean_macro_recall'] = np.mean(results['all_macro_recall'])
        results['std_macro_recall'] = np.std(results['all_macro_recall'])

        results['mean_macro_f1'] = np.mean(results['all_macro_f1'])
        results['std_macro_f1'] = np.std(results['all_macro_f1'])

        classification_report_file_name = f'{clf_name}_clf_report.json'.lower()

        classification_report_file_path = os.path.join(results_dir, classification_report_file_name)

        with open(classification_report_file_path, 'w') as file:
            json.dump(results, file, indent=4)

        ConfusionMatrixDisplay.from_predictions(all_y_test, all_y_pred)

        confusion_matrix_name = f'{clf_name}_confusion_matrix.pdf'.lower()

        img_path = os.path.join(results_dir, confusion_matrix_name)

        plt.savefig(img_path, dpi=300)
