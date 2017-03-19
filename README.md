# Kaggle - Quora Question Pairs

## Overview
This package will perform similarity analysis on Quora question pairs.

### Resulting accuracy: *~ ___*

## Usage

1. Download the GloVe embedings for Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download), unzip into a /glove directory.
        
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip
        
2. Download the kaggle data, unzip: (requires kaggle-cli)

        kg download -c quora-question-pairs

3. Run 

        python train.py

    Which will create a model.h5 and weights.h5 files.
        
## Reference
* [Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
* [GloVe embeddings](http://nlp.stanford.edu/projects/glove/)
