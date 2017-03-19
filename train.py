from __future__ import print_function

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

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

EMBEDDING_DIM = 25
GLOVE_FILE = 'glove.twitter.27B.25d.txt'

# TRAIN_DATA_FILE = "train.csv"
TRAIN_DATA_FILE = "sample_train.csv"

VALIDATION_SPLIT = 0.2

# consider outsourcing the preprocessing (tokenize + embeding) into a dictionary file)
def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # str(random.randint(0, 15))

    word_index, x_train, x_val, y_train, y_val = get_training_and_validation_sets()

    print(word_index, x_train, x_val, y_train, y_val)

    # model = make_model(labels_index, word_index)
    # train(model, x_train, x_val, y_train, y_val)
    #
    # valid_predicted_out = model.predict(x=x_val, batch_size=256)
    # evaluate(y_val, valid_predicted_out)


def get_training_and_validation_sets():
    X_raw, Y_raw = load_data_set()
    X_processed, Y_processed, word_index = tokenize_data(X_raw, Y_raw)
    x_train, x_val, y_train, y_val = split_the_data(X_processed, Y_processed)
    return word_index, x_train, x_val, y_train, y_val


def train(model, x_train, x_val, y_train, y_val):
    print("Train")
    cb = [ModelCheckpoint("weights.h5", save_best_only=True, save_weights_only=False)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=10, batch_size=256, callbacks=cb)
    try:
        os.remove("model.h5")
    except OSError:
        pass
    model.save("model.h5")

def evaluate(expected_out, predicted_out):
    expected_categories = [np.argmax(x) for x in expected_out]
    predicted_categories = [np.argmax(x) for x in predicted_out]
    cm = confusion_matrix(expected_categories, predicted_categories)
    print(cm)

def make_model(labels_index, word_index):
    embedded_sequences = make_embedding_layer(word_index)
    # Check replacing CNN to RNN with LSTM.
    # Check diff activations? softmax->tanh
    # Consider adding batch normalization
    model = Sequential([
        embedded_sequences,
        LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'),
        # Conv1D(512, 5, activation='relu'),
        # AveragePooling1D(5),
        # Conv1D(256, 5, activation='relu'),
        # AveragePooling1D(5),
        # Conv1D(128, 5, activation='relu'),
        # MaxPooling1D(5),
        # Flatten(),
        Dropout(0.3),
        # Dense(128, activation='relu'),
        Dense(len(labels_index), activation='sigmoid')
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

    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    return embedding_layer


def split_the_data(X_processed, Y_processed):
    indices = np.arange(X_processed.shape[0])
    prng.shuffle(indices)
    X_processed = X_processed[indices]
    Y_processed = Y_processed[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * X_processed.shape[0])
    x_train = X_processed[:-nb_validation_samples]
    y_train = Y_processed[:-nb_validation_samples]
    x_val = X_processed[-nb_validation_samples:]
    y_val = Y_processed[-nb_validation_samples:]

    return x_train, x_val, y_train, y_val


def tokenize_data(X_raw, Y_raw):
    texts = []
    left_texts = []
    right_texts = []
    for questions in X_raw:
        left_texts.append(questions[0])
        right_texts.append(questions[1])
        texts.append(questions[0])
        texts.append(questions[1])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    left_sequences = tokenizer.texts_to_sequences(left_texts)
    right_sequences = tokenizer.texts_to_sequences(right_texts)

    word_index = tokenizer.word_index

    left_processed = pad_sequences(left_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    right_processed = pad_sequences(right_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    for i in range(len(left_processed)):
        X_processed = [left_processed[i], right_processed[i]]

    Y_processed = to_categorical(np.asarray(Y_raw), 2) # 2 - is, or is not

    return X_processed, Y_processed, word_index


def load_data_set():
    X = []
    Y = []
    with open(TRAIN_DATA_FILE, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            print(i)
            print(line)
            text = [line[3],line[4]]
            is_duplicate = line[5] == '1'
            X.append(text)
            Y.append(is_duplicate)
    return X,Y

def get_embeddings():
    embeddings = {}
    with open(os.path.join(GLOVE_DIR, GLOVE_FILE), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    return embeddings


if __name__ == "__main__":
    main()
