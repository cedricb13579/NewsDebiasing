import argparse
import numpy as np
import pickle
import os
import sys
import json
import pdb, random
import time
from PIL import Image
from tqdm import tqdm
import nmslib

def main():
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--in_features", type=str, default="./doc2vec_features.pickle", help="Extracted document features.")
    parser.add_argument("--out_features", type=str, default="./document_feats_knn.pickle", help="Output file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Max # of processes.")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    doc_features = pickle.load(open(args.in_features, 'rb'))
    files = list(doc_features.keys())
    features = np.stack(list(doc_features.values()), axis=0).astype(np.float32)
    del doc_features
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(features)
    index.createIndex({'post': 2}, print_progress=True)
    print(f'Len feats {len(features)}')
    neighbors = index.knnQueryBatch(features, k=200, num_threads=args.num_workers)
    # Create path->neighbors lists. Each path is paired with approximate neighbors
    pickle.dump({'paths' : files, 'neighbors' : neighbors}, open(args.out_features, 'wb'))

if __name__ == '__main__':
    main()
