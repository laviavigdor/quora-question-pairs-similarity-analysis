from __future__ import print_function

from keras.callbacks import ModelCheckpoint

import model_creation_representation
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

def main():
    model, X_train, X_test, y_train, y_test = model_creation_representation.main()
    train(model, X_train, y_train)

    score = model.evaluate(X_test,y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def train(model, X_train, y_train):
    print("[train] - START")

    # validation_split - using TensorFlow recommendation, which is 1/12.
    cb = [ModelCheckpoint("weights.h5", save_best_only=True, save_weights_only=False)]
    model.fit(X_train, y_train, validation_split = 1/12, nb_epoch=10, batch_size=BATCH_SIZE, verbose=1 ,callbacks=cb)

    # run 3 more epochs using all training data (no validation_split)
    model.fit(X_train, y_train, nb_epoch=3, batch_size=BATCH_SIZE, verbose=0 ,callbacks=cb)

    try:
        os.remove(MODEL_FILE)
    except OSError:
        pass
    model.save(MODEL_FILE)
    print("[train] - END")

if __name__ == "__main__":
    main()
