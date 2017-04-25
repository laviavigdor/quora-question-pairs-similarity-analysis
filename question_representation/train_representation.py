from __future__ import print_function

from keras.callbacks import ModelCheckpoint
import gc

import numpy as np
import question_representation.model_creation_representation
import os

BASE_DIR                = '.'
GLOVE_DIR               = BASE_DIR + '/glove/'
MAX_SEQUENCE_LENGTH     = 1000
MAX_NB_WORDS            = 20000
EMBEDDING_DIM           = 25
GLOVE_FILE              = 'glove.twitter.27B.25d.txt'
BATCH_SIZE              = 256
TRAIN_DATA_FILE         = "sample_train.csv"
VALIDATION_SPLIT        = 0.2
MODEL_FILE              = "model.h5"
MODEL_FILE_EXTRA_EPOCHS = "model_extra_epochs.h5"

def main():
    model, X_train, X_test, y_train, y_test = question_representation.model_creation_representation.main()

    clear_old_files()
    train(model, X_train, y_train)
    score(model, X_test, y_test)
    gc.collect()


def clear_old_files():
    print("[clear_old_files] - START")
    try:
        os.remove(MODEL_FILE)
    except OSError:
        pass

    try:
        os.remove(MODEL_FILE_EXTRA_EPOCHS)
    except OSError:
        pass
    print("[clear_old_files] - END")

def train(model, X_train, y_train):
    print("[train] - START")

    # validation_split - using TensorFlow recommendation, which is 1/12.
    callbacks_save_best = [ModelCheckpoint(MODEL_FILE, save_best_only=True, save_weights_only=False)]
    model.fit(np.array(X_train), np.array(y_train), validation_split = 1/12, epochs=10, batch_size=BATCH_SIZE, verbose=1 ,callbacks=callbacks_save_best)


    callbacks = [ModelCheckpoint(MODEL_FILE_EXTRA_EPOCHS, save_best_only=False, save_weights_only=False)]
    # run 3 more epochs using all training data (no validation_split)
    model.fit(np.array(X_train), np.array(y_train), epochs=3, batch_size=BATCH_SIZE, verbose=0 ,callbacks=callbacks)
    print("[train] - END")

def score(model, X_test, y_test):
    score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    main()
