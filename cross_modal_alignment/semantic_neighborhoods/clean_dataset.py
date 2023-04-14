import argparse
import gc
import os, pickle
from tqdm import tqdm
import nltk
from multiprocessing import Pool

# Set up log to terminal
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from tqdm import tqdm


def trim_text(text, sentence_limit=2) -> str:
    def normalize_text(text):
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text
    text = normalize_text(text)
    # Sentence limit
    sentences = []
    sentences = nltk.tokenize.sent_tokenize(text)[:sentence_limit]
    trimmed_text = ' '.join(sentences)
    return trimmed_text

def mturk_process(dataset, temp_datafolder, trimmed_filepath, args):
    # Clean and output data to temp files
    count = 0
    for pth, info in tqdm(dataset):
        # Trim excess sentences from article
        if len(info["Input.ARTICLE_TEXT"])>0:
            info["Input.ARTICLE_TEXT"] = trim_text(info["Input.ARTICLE_TEXT"], args.sentence_limit)
        # Must dump individual samples to storage temporarily b/c I don't have enough memory to handle it + full dataset
        with open(os.path.join(temp_datafolder, f"{count}".rjust(6,'0') + ".pickle"), "wb") as o:
            pickle.dump((pth, info), o)
        count += 1

    # Free up memory
    print("Deleting dataset from memory...")
    del dataset
    gc.collect()
    print("Space freed.")

    # Merge all temp samples into one dataset again
    print("Merging trimmed data...")
    new_dataset = []
    for filename in tqdm(os.listdir(temp_datafolder)):
        if (filename.endswith(".pickle")):
            fpath = os.path.join(temp_datafolder, filename)
            with open(fpath, "rb") as infile:
                sample = pickle.load(infile)
                new_dataset.append(sample)
                os.remove(fpath)
    os.rmdir(temp_datafolder)
    print("Writing data to new file...")
    with open(trimmed_filepath, "wb") as f:
        pickle.dump(new_dataset, f)
    print("Done.")

def dataset_process(dataset, temp_datafolder, trimmed_filepath, args):
    # Clean and output data to temp files
    leanings = dataset.keys()
    topics = []
    for leaning in leanings:
        topics.extend(t for t in dataset[leaning].keys())
    topics = list(set(topics))
    for leaning in tqdm(dataset.keys(), position=0):
        for topic in tqdm(dataset[leaning].keys(), desc="topic", position=1):
            count = 0
            for sample in tqdm(dataset[leaning][topic], desc="sample", position=2, leave=False):
                # Trim excess sentences from article
                if len(sample["content_text"])>0:
                    sample["content_text"] = trim_text(sample["content_text"], args.sentence_limit)
                # Must dump individual samples to storage temporarily b/c I don't have enough memory to handle it + full dataset
                temp_filepath = os.path.join(temp_datafolder, f"{leaning}___{topic}___{count}".rjust(10,'0') + ".pickle")
                with open(temp_filepath, "wb") as o:
                    pickle.dump(sample, o)
                count += 1
                if count % args.max_samples_per_topic == 0:
                    break

    # Free up memory
    print("Deleting dataset from memory...")
    del dataset
    gc.collect()
    print("Space freed.")

    # Merge all temp samples into one dataset again
    print("Merging trimmed data...")
    new_dataset = {
        leaning: {
            topic: [] for topic in topics
        } for leaning in leanings
    }
    for filename in tqdm(os.listdir(temp_datafolder)):
        if (filename.endswith(".pickle")):
            fpath = os.path.join(temp_datafolder, filename)
            leaning, topic, _ = filename.split("___")
            with open(fpath, "rb") as infile:
                sample = pickle.load(infile)
                new_dataset[leaning][topic].append(sample)
            os.remove(fpath)
    os.rmdir(temp_datafolder)
    print("Writing data to new file...")
    with open(trimmed_filepath, "wb") as f:
        pickle.dump(new_dataset, f)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description='Trim Excess sentences from dataset metadata file.')

    parser.add_argument("--dataset", type=str, default="mturk_db.pickle", help="Dataset (.pickle) to use for Doc2Vec.")
    parser.add_argument("--sentence_limit", type=int, default=2, help="Max # sentences to use to represent each document.")
    parser.add_argument("--max_samples_per_topic", type=int, default=1000000000, help="Max # samples to save for each topic.")
    parser.add_argument("--mturk", action="store_true", help="Mturk dataset processing (probably not used).")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    # Load Initial Dataset
    print("Loading dataset...")
    dataset = pickle.load(open(args.dataset, 'rb'))
    print("Dataset loaded.")

    # Make temp storage location
    trimmed_filepath = args.dataset.replace(".pickle", "_clean.pickle")
    temp_datafolder = "./tempdatasetchunks"
    try:
        os.mkdir(temp_datafolder)
    except FileExistsError:
        pass
    print("Made temporary storage folder.")
    
    if args.mturk:
        mturk_process(dataset, temp_datafolder, trimmed_filepath, args)
    else:
        dataset_process(dataset, temp_datafolder, trimmed_filepath, args)
    
    
    
if __name__ == '__main__':
    main()
