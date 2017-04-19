import csv
import pickle
import os
from tools.UnionFind import UnionFind

DUPLICATE_SETS_FILE = "../duplicate_sets"
TRAIN_DATA_FILE = "../sample_train.csv"
# TRAIN_DATA_FILE = "sample_train.csv"

def main():
    raw_data = load_raw_data()

    if  os.path.exists(DUPLICATE_SETS_FILE):
        with open(DUPLICATE_SETS_FILE, 'rb') as pickle_file:
            duplicate_sets = pickle.load(pickle_file)
    else:
        duplicate_sets = create_duplicate_sets(raw_data)
        pickle.dump(duplicate_sets, open(DUPLICATE_SETS_FILE, "wb"))

    question_texts = create_question_texts(raw_data)
    return raw_data, duplicate_sets, question_texts

def load_raw_data():
    raw_data = []
    with open(TRAIN_DATA_FILE, "rt", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if i == 0:  # the labels line
                continue

            raw_data.append({
                'q1_id': int(line[1]),
                'q2_id': int(line[2]),
                'q1_text': line[3],
                'q2_text': line[4],
                'is_duplicate': line[5] == '1'
            })
    return raw_data

def create_question_texts(raw_data):
    question_texts = {}
    for row in raw_data:
        question_texts[row['q1_id']] = row['q1_text']
        question_texts[row['q2_id']] = row['q2_text']
    return question_texts

def create_duplicate_sets(raw_data):
    unionfind_sets = UnionFind()
    max_question_id = 0
    duplicate_sets = {}
    for row in raw_data:
        if row['is_duplicate']:
            unionfind_sets.union(row['q1_id'], row['q2_id'])
        max_question_id = max(max_question_id, row['q1_id'], row['q2_id'])

    for question_id in range(1, max_question_id):
        set_identifier = unionfind_sets[question_id]

        if set_identifier in duplicate_sets:
            duplicate_sets[set_identifier].append(question_id)
        else:
            duplicate_sets[set_identifier] = [question_id]

    return duplicate_sets


if __name__ == "__main__":
    main()
