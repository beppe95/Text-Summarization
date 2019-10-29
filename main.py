from pathlib import Path

import pandas as pd

from utils import split_in_sentences, pre_processing, get_word_embeddings

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

dataset_path = Path.cwd() / "data" / "Reviews.csv"
if __name__ == '__main__':
    dataset = pd.read_csv(dataset_path, nrows=20)
    dataset.drop_duplicates(subset=['Text'], inplace=True)
    dataset.dropna(axis=0, inplace=True)

    sentences_list = split_in_sentences(dataset['Text'])
    pre_processed_sentences = pre_processing(sentences_list)
    embeddings = get_word_embeddings('100')
