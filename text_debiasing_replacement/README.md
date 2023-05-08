To run the VisualBERT text replacement module, you need to run the VisualBertRegeneration.py file with two command arguments. The first being the mode,
either "train" or "test", the second being the number of samples used to train or test. Note that the number of samples cannot exceed the number of samples
in the available dataset. (35,000) for training and (200,000) for testing.

In order to collect the data, the training data is from the wikipedia image text dataset found here: (https://github.com/google-research-datasets/wit), and needs to be saved with the filepath,
"wikiImageText/wit_v1.train.all-1percent_sample.tsv".

In order to test the model, a saved model must be created from running the train mode first. Then three data inputs need to be aquired, a csv from the
biased word identification module, named BiasedWordIdentificationTableRun1.csv. Next a set of test images corresponing to the text from the biased word identification csv, placed in a filepath:
(./new_imgs/new_imgs). Finally a wiki-news-300d-1M-subword.vec wordVector model obtained from (https://fasttext.cc/docs/en/english-vectors.html) placed in a filepath "./wordVecModel/model.txt. Then the test mode can run.


