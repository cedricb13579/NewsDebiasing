import csv
import pickle
import random
import nltk
import sys
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

# csv.field_size_limit(sys.maxsize)
# Above line breaks on my machine, changing to try except based on
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    maxInt = int(sys.maxsize/10)
d2v = Doc2Vec.load('./doc2vec_model.gensim')
word_to_vector = d2v.wv
# word_to_index = {word : idx for idx, word in enumerate(word_to_vector.index2entity)}
word_to_index = {word : idx for idx, word in enumerate(word_to_vector.index_to_key)}
del d2v
word_to_index['<!START!>'] = len(word_to_index)
word_to_index['<!END!>'] = len(word_to_index)

# Load KNN - THIS IS PUT INTO SHARED MEMORY FOR MULTIPROCESSING
neighbors = pickle.load(open('./document_feats_knn.pickle', 'rb'))
_neighbors_pths2idx = {v : k for k,v in enumerate(neighbors['paths'])} # SHARED!
_neighbors_idxs, _neighbors_dists = zip(*neighbors['neighbors'])
_neighbors_idxs = np.stack([np.pad(a.ravel(), (0, 200 - a.size), 'constant', constant_values=0) for a in _neighbors_idxs], axis=0) # SHARED
_neighbors_dists = np.stack([np.pad(a.ravel(), (0, 200 - a.size), 'constant', constant_values=0) for a in _neighbors_dists], axis=0)  # SHARED
_neighbors_pths = neighbors['paths'] # SHARED
del neighbors  # Free python object - use only Numpy variants

def get_db(datapath):
    # db = pickle.load(open('complete_db.pickle', 'rb'))
    f = open(datapath, 'rb')
    db = pickle.load(f)
    f.close()
    orig_paths_and_text = []
    for politics, issues in db.items():
        for issue, items in issues.items():
            for item in items:
                # if len(item['content_text']) > 500:
                # I changed this because I already cut down articles to 2 sentences
                if len(item['content_text']) > 10:
                    orig_paths_and_text.append(
                        (item['local_path'], item['content_text']))
    random.shuffle(orig_paths_and_text)
    return orig_paths_and_text
    
def convert_text(tup, sentence_limit=2):
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
    tokenized_text = preprocess_string(text, CUSTOM_FILTERS)
    tokenized_text = list(filter(lambda x: x in word_to_index, tokenized_text))
    return (pth, tokenized_text)