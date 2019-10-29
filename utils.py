from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

stop_words = stopwords.words('english')


def split_in_sentences(dataset_text: pd.Series) -> list:
    """

    :param dataset_text:
    :return:
    """
    sentences = []
    for s in dataset_text:
        sentences.append(sent_tokenize(s))
    return [text for sentence in sentences for text in sentence]


# noinspection PyTypeChecker
def get_word_embeddings(input_dim: str) -> defaultdict:
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
