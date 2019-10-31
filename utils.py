from collections import defaultdict
from pathlib import Path
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

stop_words = stopwords.words('english')

possible_inputs = ['50', '100', '200', '300']


def split_in_sentences(dataset_text: pd.Series) -> list:
    """

    :param dataset_text:
    :return:
    """
    sentences = []
    for s in dataset_text:
        sentences.append(sent_tokenize(s))
    return [text for sentence in sentences for text in sentence]


def remove_html_tag(sentences: list) -> list:
    """

    :param sentences:
    :return:
    """
    return [BeautifulSoup(sentence, features='html.parser').text for sentence in sentences]


def remove_stopwords(sentences: list) -> str:
    """

    :param sentences:
    :return:
    """
    no_stop_sen = " ".join([sentence for sentence in sentences if sentences not in stop_words])
    return no_stop_sen


def pre_processing(sentences: list) -> list:
    """

    :param sentences:
    :return:
    """
    processed_sentences = pd.Series(sentences).replace("[^a-zA-Z]", " ")
    processed_sentences = [sentence.lower() for sentence in processed_sentences]
    return [remove_stopwords(p_sent.split()) for p_sent in processed_sentences]


# noinspection PyTypeChecker
def get_word_embeddings(input_dim: int) -> defaultdict:
    """

    :param input_dim:
    :return:
    """
    with open(Path.cwd() / "glove" / ("glove.6B." + input_dim + "d.txt"), encoding='utf-8') as embedding_file:
        word_embeddings = defaultdict()
        for line in embedding_file:
            line_values = line.split()
            word = line_values[0]
            coefficients = np.asarray(line_values[1:], dtype='float32')
            word_embeddings[word] = coefficients

        return word_embeddings


def ask_embedding_dim():
    """

    :return:
    """
    choice = input('Choose an embedding dimensionality.\nPossible values are: 50, 100, 200, 300.')
    if choice not in possible_inputs:
        warn('Read carefully, your choice is not allowed!')
        return ask_embedding_dim()
    return choice


def make_sentences_vectors(pre_processed_sentences: list, embeddings: defaultdict, embedding_dim: int) -> list:
    """

    :param pre_processed_sentences:
    :param embeddings:
    :param embedding_dim:
    :return:
    """
    sentences_vectors = []
    for sent in pre_processed_sentences:
        if len(sent) > 0:
            v = sum([embeddings.get(word, np.zeros(embedding_dim, )) for word in sent.split()]) / (
                    len(sent.split()) + 0.001)
        else:
            v = np.zeros(embedding_dim, )
        sentences_vectors.append(v)
    return sentences_vectors


def make_similarity_matrix(sentences: list, sents_vects: list, embedding_dim: int) -> np.zeros:
    """

    :param sentences:
    :param sents_vects:
    :param embedding_dim:
    :return:
    """
    sent_len = len(sentences)

    similarity_matrix = np.zeros(shape=(sent_len, sent_len))

    for i in range(sent_len):
        for j in range(sent_len):
            if i != j:
                similarity_matrix[i, j] = cosine_similarity(sents_vects[i].reshape(1, embedding_dim),
                                                            sents_vects[j].reshape(1, embedding_dim))
    return similarity_matrix


def apply_pagerank(sim_mat: np.zeros) -> dict:
    """

    :param sim_mat:
    :return:
    """
    nx_graph = nx.from_numpy_array(sim_mat)
    return nx.pagerank(nx_graph)


def ask_top_n_sentences_to_extract() -> int:
    """

    :return:
    """
    choice = int(input('Choose the number of sentences to extract for summary generation'))
    if choice < 1:
        warn('You must extract at least one sentence!')
        return ask_top_n_sentences_to_extract()
    return choice


def extract_sentences(n: int, sentences: list, scores: dict) -> list:
    """

    :param n:
    :param sentences:
    :param scores:
    :return:
    """
    ranked_sentences = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)
    return ranked_sentences[:n]
