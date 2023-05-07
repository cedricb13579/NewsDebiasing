## Running
1. Follow the installation instructions from https://github.com/CLT29/semantic_neighborhoods
2. Download the Politics dataset (https://people.cs.pitt.edu/~chris/politics/)
3. Trim and clean dataset:
   - ```python clean_dataset.py --dataset  <path/to/dataset_metadata.pickle> --max_samples_per_topic 5000```
4. Train Doc2Vec Space:
   - ```python train_doc2vec.py --dataset <path/to/dataset_metadata_clean_5000.pickle> --split_paths <path/to/train_test_paths.pickle> --sentence_limit 2```
   - ```python extract_doc2vec_vectors.py --dataset <path/to/dataset_metadata_clean_5000.pickle> --split_paths <path/to/train_test_paths.pickle>```
   - ```python knn_document_features.py --in_features doc2vec_features.pickle --out_features document_feats_knn.pickle```
5. Set flags to 'True' and run the Cross_Modal_Alignment.ipynb notebook
