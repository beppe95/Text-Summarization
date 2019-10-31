from pathlib import Path

import pandas as pd

from utils import split_in_sentences, remove_html_tag, pre_processing, get_word_embeddings, ask_embedding_dim, \
    make_sentences_vectors, make_similarity_matrix, apply_pagerank, ask_top_n_sentences_to_extract, extract_sentences

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

dataset_path = Path.cwd() / "data" / "Reviews.csv"
if __name__ == '__main__':
    dataset = pd.read_csv(dataset_path, nrows=100)
    dataset.drop_duplicates(subset=['Text'], inplace=True)
    dataset.dropna(axis=0, inplace=True)

    sentences_list = split_in_sentences(dataset['Text'])
    sentences_list = remove_html_tag(sentences_list)

    pre_processed_sentences = pre_processing(sentences_list)

    embedding_dimensionality = ask_embedding_dim()
    embeddings = get_word_embeddings(embedding_dimensionality)

    sents_vects = make_sentences_vectors(pre_processed_sentences, embeddings, int(embedding_dimensionality))

    similarity_matrix = make_similarity_matrix(sentences_list, sents_vects, int(embedding_dimensionality))

    pagerank_scores = apply_pagerank(similarity_matrix)

    number_sentences_to_extract = ask_top_n_sentences_to_extract()

    for ex_sent in extract_sentences(number_sentences_to_extract, sentences_list, pagerank_scores):
        print(ex_sent, "\n")
