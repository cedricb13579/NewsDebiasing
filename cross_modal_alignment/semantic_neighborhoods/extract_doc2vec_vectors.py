import argparse

import os, pdb, sys, glob, time, re, pickle, random, gensim, json, unicodedata
from tqdm import tqdm
import numpy as np
import nltk
from multiprocessing import Pool
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum
from tqdm import tqdm

# Had to change to a container layout b/c global d2v was refusing to cooperate
class Container:
    def __init__(self, args):
        self.d2v:Doc2Vec = None
        self.args = args

    def process_text(self, tup, sentence_limit=2):
        def normalize_text(text):
            text = text.encode('ascii', errors='replace').decode('ascii')
            return text
        pth, text = tup
        text = normalize_text(text)
        # Sentence limit
        sentences = []
        for sentence in nltk.tokenize.sent_tokenize(text)[:sentence_limit]:
            sentences.append(sentence)
        text = ' '.join(sentences)
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short]
        tokenized_text = list(preprocess_string(text, CUSTOM_FILTERS))
        if tokenized_text:
            features = self.d2v.infer_vector(tokenized_text)
            return (pth, features)
        else:
            return (pth, None)

    def process(self):
        args = self.args

        dataset = pickle.load(open(args.dataset, 'rb'))
        print("Loaded Dataset.")
        train_pths = set(pickle.load(open(args.split_paths, 'rb'))['train'])
        print("Loaded Split Paths.")
        dataset = [(pth, info["Input.ARTICLE_TEXT"]) for pth, info in dataset if pth in train_pths and len(info["Input.ARTICLE_TEXT"])>0]
        print(f'Len deduped dataset = {len(dataset)}')
        self.d2v = Doc2Vec.load(args.doc2vec_model)
        pool = Pool(processes=args.num_workers)
        pth_to_features = list(tqdm(pool.imap_unordered(self.process_text, dataset, chunksize=1), total=len(dataset)))
        pool.close()
        pth_to_features = dict([(pth, features.ravel()) for pth, features in pth_to_features if features is not None])
        pickle.dump(pth_to_features, open(args.output_path, 'wb'))

def main():
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--dataset", type=str, default="mturk_db.pickle", help="Dataset (.pickle) to use for Doc2Vec.")
    parser.add_argument("--split_paths", type=str, default="train_test_paths.pickle", help="Split file (.pickle) to define which images are associated with each split.")
    parser.add_argument("--doc2vec_model", type=str, default="./doc2vec_model.gensim", help="Path to Doc2Vec model.")
    parser.add_argument("--output_path", type=str, default="doc2vec_features.pickle", help="Output file path (.pickle)")
    parser.add_argument("--num_workers", type=int, default=4, help="Max # of processes.")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    processor = Container(args)
    processor.process()
    
if __name__ == '__main__':
    main()
