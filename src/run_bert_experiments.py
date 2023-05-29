import os
import torch
import numpy as np
import torch.nn.functional as f

from corpus_utils import read_corpus
from nlp_utils import preprocessing_v2
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, AlbertTokenizer, BertTokenizer, RobertaTokenizer, DebertaTokenizer
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, DistilBertForSequenceClassification, \
    RobertaForSequenceClassification, DebertaForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import Dataset
from bert_helper import tokenize_resume, compute_metrics_classification
from src.evaluation_utils import compute_evaluation_measures, compute_means_std_eval_measures


if __name__ == '__main__':

    corpus_path = '../resumes_corpus'

    n_total = 200

    # model_name = 'distil_bert_base'
    model_name = 'albert_base'
    # model_name = 'albert_large'
    # model_name = 'bert_base'
    # model_name = 'bert_large'
    # model_name = 'roberta_base'
    # model_name = 'roberta_large'
    # model_name = 'deberta_base'
    # model_name = 'deberta_large'

    num_epochs = 1
    # num_epochs = 5
    # num_epochs = 10
    # num_epochs = 20

    n_splits = 5

    batch_size = 8
    max_len = 512

    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    fp16 = False
    optim = 'adamw_torch'

    if model_name in ['bert_large', 'deberta_large', 'roberta_large']:
        fp16 = True
        batch_size = 4

    results_dir = f'../results/bert/{model_name}/{num_epochs}'

    os.makedirs(results_dir, exist_ok=True)

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

    print(f'\nLabels distribution: {labels_distribution}')

    label_encoder = LabelEncoder()

    y_labels = label_encoder.fit_transform(labels)

    print(f'\nLabels Mappings: {label_encoder.classes_}')

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    print(f'\nModel Name: {model_name}')

    model_path = None

    if model_name == 'distil_bert_base':
        model_path = 'distilbert-base-uncased'
    elif model_name == 'albert_base':
        model_path = 'albert-base-v2'
    elif model_name == 'albert_large':
        model_path = 'albert-large-v2'
    elif model_name == 'bert_base':
        model_path = 'bert-base-uncased'
    elif model_name == 'bert_large':
        model_path = 'bert-large-uncased'
    elif model_name == 'roberta_base':
        model_path = 'roberta-base'
    elif model_name == 'roberta_large':
        model_path = 'roberta-large'
    elif model_name == 'deberta_base':
        model_path = 'microsoft/deberta-base'
    elif model_name == 'deberta_large':
        model_path = 'microsoft/deberta-large'
    else:
        print('ERROR. Model Name Invalid!')
        exit(-1)

    print(f'\nConfigurations: {num_epochs} -- {batch_size}')

    if model_name == 'distil_bert_base':
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    elif model_name in ['albert_base', 'albert_large']:
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
    elif model_name in ['bert_base', 'bert_large']:
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif model_name in ['roberta_base', 'roberta_large']:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    elif model_name in ['deberta_base', 'deberta_large']:
        tokenizer = DebertaTokenizer.from_pretrained(model_path)
    else:
        print('ERROR. Model Name Invalid!')
        exit(-1)

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

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

        y_train = f.one_hot(y_train.to(torch.int64), num_classes=num_classes)
        y_test = f.one_hot(y_test.to(torch.int64), num_classes=num_classes)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, shuffle=True, random_state=42)

        train_dict = {'text': X_train, 'label': y_train}
        valid_dict = {'text': X_valid, 'label': y_valid}
        test_dict = {'text': X_test, 'label': y_test}

        train_dataset = Dataset.from_dict(train_dict)
        valid_dataset = Dataset.from_dict(valid_dict)
        test_dataset = Dataset.from_dict(test_dict)

        encoded_train_dataset = train_dataset.map(lambda x: tokenize_resume(x, tokenizer, max_len),
                                                  batched=True, batch_size=batch_size)
        encoded_valid_dataset = valid_dataset.map(lambda x: tokenize_resume(x, tokenizer, max_len),
                                                  batched=True, batch_size=batch_size)
        encoded_test_dataset = test_dataset.map(lambda x: tokenize_resume(x, tokenizer, max_len),
                                                batched=True, batch_size=batch_size)

        print(f'\n  Folder {k + 1} - {len(X_train)} - {len(X_valid)} - {len(X_test)}')

        model = None

        if model_name == 'distil_bert_base':
            model = DistilBertForSequenceClassification.from_pretrained(model_path,
                                                                        num_labels=num_classes)
        elif model_name in ['bert_base', 'bert_large']:
            model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
        elif model_name in ['albert_base', 'albert_large']:
            model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
        elif model_name in ['roberta_base', 'roberta_large']:
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
        elif model_name in ['deberta_base', 'deberta_large']:
            model = DebertaForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)

        training_args = TrainingArguments(output_dir='training', logging_strategy='epoch',
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          gradient_checkpointing=gradient_checkpointing,
                                          fp16=fp16, optim=optim, weight_decay=0.01, eval_steps=100,
                                          logging_steps=100, learning_rate=5e-5,
                                          evaluation_strategy='epoch',
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          num_train_epochs=num_epochs, save_total_limit=2,
                                          save_strategy='epoch', load_best_model_at_end=True,
                                          metric_for_best_model='f1_macro', greater_is_better=True,
                                          report_to=['none'])

        trainer = Trainer(model=model, args=training_args, train_dataset=encoded_train_dataset,
                          eval_dataset=encoded_valid_dataset,
                          compute_metrics=compute_metrics_classification,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

        trainer.train()

        y_pred, _, _ = trainer.predict(encoded_test_dataset)

        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        y_test = [int(y.item()) for y in y_test]
        y_pred = [int(y.item()) for y in y_pred]

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        compute_evaluation_measures(y_test, y_pred, results_dict)

    compute_means_std_eval_measures(model_name, all_y_test, all_y_pred, results_dict, results_dir)
