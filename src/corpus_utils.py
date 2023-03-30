import os
import pandas as pd
import sys
import random

from tqdm import tqdm


def read_corpus(corpus_path: str, num_examples: int = -1) -> pd.DataFrame:
    docs_names = os.listdir(corpus_path)
    txt_files = [d for d in docs_names if d.endswith('.txt')]
    if num_examples > -1:
        random.shuffle(txt_files)
        txt_files = txt_files[:num_examples]
    resumes = []
    labels = []
    with tqdm(total=len(txt_files), file=sys.stdout, colour='red', desc='  Loading ') as pbar:
        for txt_file_name in txt_files:
            txt_file_path = os.path.join(corpus_path, txt_file_name)
            lab_file_path = os.path.join(corpus_path, txt_file_name.replace('.txt', '.lab'))
            with open(txt_file_path, 'r') as file:
                txt_content = file.read().strip()
            with open(lab_file_path, 'r') as file:
                labels_content = file.read().lower().strip().split()
            if len(labels_content) > 0:
                resumes.append(txt_content)
                labels.append(labels_content)
            pbar.update(1)
    corpus_dict = {
        'resume': resumes,
        'label': labels
    }
    corpus_df = pd.DataFrame.from_dict(corpus_dict)
    return corpus_df
