import pickle
import os
import argparse
from multiprocessing import Pool, Manager
import time
from tqdm import tqdm

from PIL import Image
from io import BytesIO
import base64

# new_dataset = {}

def init_pool(dictX):
    # function to initial global dictionary
    global new_dataset
    new_dataset = dictX

# From https://github.com/OFA-Sys/OFA
def encode_img(file_name):
    img = Image.open(file_name) # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str

def decode_img(img_str):
    imgdata = base64.b64decode(img_str)
    img = Image.open(BytesIO(imgdata))
    img = img.convert('RGB')
    return img

def to_gdict(sample, base_path=""):
    new_dataset.setdefault(sample["local_path"], encode_img(os.path.join("E://Research//Political_Image_Debiasing//dataset", sample["local_path"])))

    # new_dataset.setdefault(sample["local_path"], encode_img(os.path.join(base_path, sample["local_path"])))

    # new_dataset[sample["local_path"]] = encode_img(os.path.join(base_path, sample["local_path"]))

def main():
    global new_dataset
    parser = argparse.ArgumentParser(description='Encodes images as base64 strings')

    parser.add_argument("--dataset", type=str, default="dataset_metadata_clean_5000.pickle", help="Dataset (.pickle) to use for finding images.")
    parser.add_argument("--img_base_path", type=str, help="Folder containing images.")
    parser.add_argument("--output_path", type=str, default="doc2vec_model.gensim", help="Output file path (.gensim)")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []

    # Load Initial Dataset
    print("Loading dataset...")
    f = open(args.dataset, 'rb')
    dataset = pickle.load(f)
    f.close()
    print("Dataset loaded.")

    new_datafile = {} # maps img path to img string
    for leaning in tqdm(dataset.keys(), position=0):
        for topic in tqdm(dataset[leaning].keys(), desc="topic", position=1):
            start = time.time()
            samples = dataset[leaning][topic]
            with Manager() as manager:
                new_dataset = manager.dict()
                pool = Pool(initializer=init_pool, initargs=(new_dataset,), processes=6) # initial global dictionary
                tqdm(pool.imap(to_gdict, samples, chunksize=1), total=len(samples))
                # tqdm(pool.starmap(to_gdict, zip(samples, [args.img_base_path]*len(samples)), chunksize=1), total=len(dataset[leaning][topic]))
                pool.close()
                pool.join()
                stop = time.time()
                new_datafile.update(new_dataset)
                print(f"Datafile with {len(new_datafile)} entries...")
                print('Done in {:4f}'.format(stop-start))
                # print(globalDict)
            # pool = Pool(processes=4)
            # tqdm(pool.imap(to_dict, dataset[leaning][topic], chunksize=1), total=len(dataset[leaning][topic]))
            # pool.close()
            # for sample in tqdm(dataset[leaning][topic], desc="sample", position=2, leave=False):
            #     new_datafile[sample["local_path"]] = encode_img(os.path.join(args.img_base_path, sample["local_path"]))
    
    print(f"Datafile with {len(new_datafile)} entries...")
    with open(args.output_path, "wb") as f:
        pickle.dump(new_datafile, f)

    
if __name__ =="__main__":
    main()