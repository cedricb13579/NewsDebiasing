import argparse

import os, pdb, sys, glob, time, re, pickle, random, gensim, json, unicodedata
from tqdm import tqdm
import numpy as np
import nltk
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum

# Set up log to terminal
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from tqdm import tqdm


def convert_text(text, sentence_limit=2):
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    text = normalize_text(text)
    # Sentence limit
    sentences = []
    for sentence in nltk.tokenize.sent_tokenize(text)[:sentence_limit]:
        sentences.append(sentence)
    text = ' '.join(sentences)
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
    tokenized_text = list(preprocess_string(text, CUSTOM_FILTERS))
    return tokenized_text

def main():
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--dataset", type=str, default="mturk_db.pickle", help="Dataset (.pickle) to use for Doc2Vec.")
    parser.add_argument("--split_paths", type=str, default="train_test_paths.pickle", help="Split file (.pickle) to define which images are associated with each split.")
    parser.add_argument("--sentence_limit", type=int, default=2, help="Max # sentences to use to represent each document.")
    parser.add_argument("--output_path", type=str, default="doc2vec_model.gensim", help="Output file path (.gensim)")
    parser.add_argument("--num_workers", type=int, default=4, help="Max # of processes for tokenizing and doc2vec.")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    dataset = pickle.load(open(args.dataset, 'rb'))
    print("Loaded Dataset.")
    train_pths = set(pickle.load(open(args.split_paths, 'rb'))['train'])
    print("Loaded Split Paths.")
    pool = Pool(processes=args.num_workers)
    all_train_text = [info["Input.ARTICLE_TEXT"] for pth, info in dataset if pth in train_pths and len(info["Input.ARTICLE_TEXT"])>0]
    documents = list(tqdm(pool.imap(convert_text, all_train_text, chunksize=1), total=len(all_train_text)))
    pool.close()
    print("Tokenized Documents.")
    documents = [d for d in documents if d]
    random.shuffle(documents)
    print("Shuffled Documents.")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    print('Starting training on {} documents'.format(len(documents)))
    # WE REPORT USING 20 EPOCHS IN PAPER - BUT USE 50 ON SMALLER DATASETS - MAY NEED ADJUSTING DEPENDING ON DATASET SIZE
    d2v = Doc2Vec(documents=documents, vector_size=200, workers=args.num_workers, epochs=50, window=20, min_count=20)
    # d2v.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2v.save(args.output_path)
    print('Finished training')
if __name__ == '__main__':
    main()
