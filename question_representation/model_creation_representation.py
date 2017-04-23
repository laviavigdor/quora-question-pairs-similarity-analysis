from keras.models import Sequential
from keras.layers import Dropout, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from sklearn import model_selection

import question_representation.preprocess_representation
import os
import numpy as np

BASE_DIR                 = '..'
GLOVE_DIR                = BASE_DIR + '/glove/'
MAX_WORDS_IN_SENTENCE    = 40
EMBEDDING_DIM_FOR_WORD   = 25
GLOVE_VECTOR_SIZE        = MAX_WORDS_IN_SENTENCE * EMBEDDING_DIM_FOR_WORD
MAX_NUM_OF_WORDS         = 20000 #the top 20,000 most commonly occuring words in the dataset
GLOVE_FILE               = 'glove.twitter.27B.25d.txt'
TEST_SIZE                = 0.2

import preprocess

def main():
    raw_data, duplicate_sets, question_texts = question_representation.preprocess_representation.main()
    model, word_index, ids_sequences = create_model(len(duplicate_sets), question_texts)
    X_train, X_test, y_train, y_test = get_training_and_validation_sets(duplicate_sets, ids_sequences, len(duplicate_sets))
    return model, X_train, X_test, y_train, y_test

def get_training_and_validation_sets(duplicate_sets, ids_sequences, number_of_categories):
    print("[get_training_and_validation_sets] - START")
    X = []
    y = []

    for question_id, sequence in ids_sequences.items():
        X.append(sequence)
        y_vector = []
        for set in duplicate_sets.values():
            if question_id in set:
                y_vector.append(1)
            else:
                y_vector.append(0)
        y.append(y_vector)
    # random_state (WITH ANY NUMBER) will split always the same for a given data
    # this is in case we want to reproduce result
    # if we always want different split, then remove it
    print("[get_training_and_validation_sets] - END")
    return model_selection.train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

def create_model(number_of_categories, question_texts):
    print("[create_model] - START")
    ids_sequences, word_index = tokenize_data(question_texts)
    embedding_layer = make_embedding_layer(word_index)

    model = Sequential()
    model.add(embedding_layer)

    # return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    # TODO - TEST WITH 256/512/1024
    model.add(LSTM(512, return_sequences=True, activation='tanh', inner_activation='hard_sigmoid',
                   input_shape=(GLOVE_VECTOR_SIZE,)))

    model.add(Dropout(0.2))

    # TODO - TEST WITH 256/512/1024
    model.add(LSTM(512, return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(number_of_categories))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    print("[create_model] - END")
    return model, word_index, ids_sequences

def make_embedding_layer(word_index):
    print("[make_embedding_layer] - START")

    words_vector = get_embeddings()
    num_of_words = min(MAX_NUM_OF_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_of_words, EMBEDDING_DIM_FOR_WORD))

    for word, index in word_index.items():
        if index >= num_of_words:
            continue
        embedding_vector = words_vector.get(word)
        if embedding_vector is not None: # in case we dont find a word, its vector will be [0,...,0]
            embedding_matrix[index] = embedding_vector

    # create the embedding layer from embedding matrix
    # All that the Embedding layer does is to map the integer inputs to the vectors found at the
    # corresponding index in the embedding matrix, i.e. the sequence [1, 2] would be converted to [embeddings[1], embeddings[2]].
    embedding_layer = Embedding(input_dim=num_of_words,
                                output_dim=EMBEDDING_DIM_FOR_WORD,
                                weights=[embedding_matrix],
                                input_length=MAX_WORDS_IN_SENTENCE,
                                trainable=False) # set to be frozen, the embedding vectors will not be updated during training.

    print("[make_embedding_layer] - END")
    return embedding_layer

def get_embeddings():
    word_vec = {}
    with open(os.path.join(GLOVE_DIR, GLOVE_FILE), 'r',  encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            current_word_glove_vector = np.asarray(values[1:], dtype='float32')
            word_vec[word] = current_word_glove_vector

    print('Found %s word vectors.' % len(word_vec))
    return word_vec

def tokenize_data(sentences):
    print("[tokenize_data] - START")

    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
    tokenizer.fit_on_texts(sentences.values())
    sequences = tokenizer.texts_to_sequences(sentences.values())
    word_index = tokenizer.word_index

    padded_sequences = pad_sequences(sequences, maxlen=MAX_WORDS_IN_SENTENCE, padding = 'post')

    if len(sentences.values()) != len(padded_sequences):
        raise Exception('[tokenize_data] - sentences amount does not match sequences amount')
    else:
        # amount is matching, so we can map sentence id to its sequence
        ids_sequences = dict(zip(sentences.keys(), padded_sequences))

    print('Found %s unique tokens.' % len(word_index))
    print("[tokenize_data] - END")
    return ids_sequences, word_index

if __name__ == "__main__":
    main()
