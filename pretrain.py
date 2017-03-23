
from keras.models import Sequential
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,  Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding, LSTM

from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

import os
import csv
import numpy as np
from numpy.random import RandomState
prng = RandomState(1234567890)

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove/'

MAX_SEQUENCE_LENGTH = 40 # ?
MAX_NB_WORDS = 200000

EMBEDDING_DIM = 25
GLOVE_FILE = 'glove.twitter.27B.25d.txt'

import preprocess

def main():
    raw_data, raw_data, duplicate_sets, question_texts = preprocess.main()
    number_of_categories = len(duplicate_sets)

    tokenized_sentences, word_index = tokenize_data(question_texts.values())
    # Y_processed = to_categorical(np.asarray(Y_raw), 2)

    embedded_sequences = make_embedding_layer(word_index)
    model = make_model(embedded_sequences, number_of_categories)


def make_model(embedded_sequences, number_of_categories):
    model = Sequential([
        embedded_sequences,
        Dense(8192, activation='relu'),
        # Dropout(0.2),
        Dense(number_of_categories, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def make_embedding_layer(word_index):
    embeddings = get_embeddings()
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(input_dim=nb_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer

def get_embeddings():
    embeddings = {}
    with open(os.path.join(GLOVE_DIR, GLOVE_FILE), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings



def tokenize_data(sentences):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    word_index = tokenizer.word_index
    tokenized_sentences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return tokenized_sentences, word_index


if __name__ == "__main__":
    main()
