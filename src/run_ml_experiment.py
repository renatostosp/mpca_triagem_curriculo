from corpus_utils import read_corpus
from nlp_utils import preprocessing
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import classification_report


if __name__ == '__main__':

    corpus_path = 'D:\\Hilario\\Pesquisa\\Experimentos\\Renato\\corpus\\resumes_corpus'

    n_splits = 10

    n_total = -1

    max_features = 100

    vectorizer_opt = 'binary'
    # vectorizer_opt = 'count'
    # vectorizer_opt = 'tf_idf'

    print('\nLoading Corpus\n')

    corpus_df = read_corpus(corpus_path)

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

    print('\nExample Encoded:')
    print(f'  Resume: {X_resumes[-1]}')
    print(f'  Label: {y_labels[-1]}')

    classifiers = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        # 'KNN': KNeighborsClassifier(weights='uniform'),
        # 'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
        # 'RandomForest': RandomForestClassifier(n_estimators=1000, class_weight='balanced'),
        # 'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=1000, class_weight='balanced'),
        # 'xgboost': XGBClassifier(n_estimators=1000),
        # 'lgbm': LGBMClassifier(n_estimators=1000, class_weight='balanced'),
        # 'svc': SVC(class_weight='balanced'),
        # 'CatBoostClassifier': CatBoostClassifier(n_estimators=1000, verbose=False)
    }

    print('\n\n------------Evaluations------------\n')

    for clf_name, clf_base in classifiers.items():

        print(f'\n  Classifier: {clf_name}')

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X_resumes, y_labels):

            classifier = clone(clf_base)

            X_train, X_test = X_resumes[train_index], X_resumes[test_index]

            y_train, y_test = y_labels[train_index], y_labels[test_index]

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            report_folder = classification_report(y_test, y_pred, zero_division=0, output_dict=False)

            print(report_folder)

