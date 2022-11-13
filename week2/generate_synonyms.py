import argparse
from fasttext import load_model
import os


directory = r'/workspace/datasets/fasttext/'
model_file = 'title_model.bin'
top_words_file = 'top_words.txt'
outputs_file = 'synonyms.csv'
model = load_model(os.path.join(directory, model_file))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--threshold", default=0.75, type=float, help="Threshold similarity score for nearest neighbor.")
args = parser.parse_args()

with open(os.path.join(directory, top_words_file), 'r') as f:
    with open(os.path.join(directory, outputs_file), 'w') as w:
        for line in f:
            word = line.strip()
            scores = model.get_nearest_neighbors(word)
            neighbors = []
            for score, nn in scores:
                if score >= args.threshold:
                    neighbors.append(nn)
            w.write(','.join([word] + neighbors) + '\n')
