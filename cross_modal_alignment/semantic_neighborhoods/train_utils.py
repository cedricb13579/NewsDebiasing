import csv
import pickle
import random
import nltk
import sys
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, strip_non_alphanum
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

csv.field_size_limit(sys.maxsize)
d2v = Doc2Vec.load('./doc2vec_model.gensim')
word_to_vector = d2v.wv
word_to_index = {word : idx for idx, word in enumerate(word_to_vector.index2entity)}
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

def get_db():
    db = pickle.load(open('complete_db.pickle', 'rb'))
    orig_paths_and_text = []
    for politics, issues in db.items():
        for issue, items in issues.items():
            for item in items:
                if len(item['content_text']) > 500:
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

def get_doc2vec_neighbors(pth, transform, orig_img):
    assert(pth in _neighbors_pths2idx)    
    root_idx = _neighbors_pths2idx[pth]
    root_neighbor_idxs = _neighbors_idxs[root_idx, :]
    root_neighbor_dists = _neighbors_dists[root_idx, :]
    non_dupe_idxs = np.nonzero(np.abs(root_neighbor_dists) > 0)[0].tolist()
    neighbor_imgs = []
    neighbor_pths = []
    non_dupe_idxs = [ndi for ndi in non_dupe_idxs if _neighbors_pths[root_neighbor_idxs[ndi]] in MyDataset.train_set and _neighbors_pths[root_neighbor_idxs[ndi]] in MyDataset.result_db]
    # Choose one randomly from first N (here we use N=10 most similar neighbors - worked slightly in later tests) but you can also adjust to all 200
    non_dupe_idxs = non_dupe_idxs[:10]
    random.shuffle(non_dupe_idxs)
    # IF YOU WANT - YOU CAN USE MORE THAN ONE NEIGHBOR
    for ndi in non_dupe_idxs:
        neighbor_idx = root_neighbor_idxs[ndi]
        neighbor_pth = _neighbors_pths[neighbor_idx]
        if neighbor_pth not in MyDataset.train_set or root_idx == neighbor_idx or neighbor_pth not in MyDataset.result_db:
            continue
        try:
            img = transform(Image.open(neighbor_pth).convert('RGB'))
        except:
            continue
        neighbor_imgs.append(img)
        neighbor_pths.append(neighbor_pth)
    # USING 1 NEIGHBOR
    return (neighbor_imgs[0], neighbor_pths[0])