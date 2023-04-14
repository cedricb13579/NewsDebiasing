import argparse
import gc
import os, pickle
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

# Set up log to terminal
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from tqdm import tqdm

def data_to_csv(datapath):
    data = []
    with open(datapath, "rb") as f:
        dataset = pickle.load(f)
        for leaning in tqdm(dataset.keys(), position=0):
            for topic in tqdm(dataset[leaning].keys(), desc="topic", position=1):
                for sample in tqdm(dataset[leaning][topic], desc="sample", position=2, leave=False):
                    source = os.path.dirname(sample["local_path"]).split("/")[-1]
                    data.append((sample["local_path"], source, sample["content_text"]))
        df = pd.DataFrame(data, columns=["img_filepath", "website", "text"])
        df.to_csv("dataset.csv")

def main():
    parser = argparse.ArgumentParser(description='Trim Excess sentences from dataset metadata file.')
    parser.add_argument("--dataset", type=str, default="dataset_metadata_clean.pickle", help="Dataset (.pickle).")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    data_to_csv(args.dataset)

if __name__ == "__main__":
    main()
