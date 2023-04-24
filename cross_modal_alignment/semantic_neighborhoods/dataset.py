import pickle
import random
import torch
import numpy as np

from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from torchvision import transforms

from train_utils import get_db, convert_text
from train_utils import _neighbors_pths2idx, word_to_index, _neighbors_idxs, _neighbors_pths, _neighbors_dists

import os

class MyDataset(torch.utils.data.Dataset):
    result_db = {}
    train_set = []
    val_set = []
    def __init__(self, mode, datapath):
        super().__init__()
        self.base_path = "E:/Research/Political_Image_Debiasing/dataset/"
        if not MyDataset.result_db:
            # train_test_dict = pickle.load(open('train_test_pths.pickle', 'rb'))
            train_test_dict = pickle.load(open('train_test_paths.pickle', 'rb'))
            train_set = train_test_dict['train']
            val_set = set(random.sample(train_set, len(train_test_dict['test'])))
            train_set = [pth for pth in train_set if pth not in val_set]
            MyDataset.train_set = list(train_set)
            MyDataset.test_set = list(val_set)
            complete_db = [(pth, txt[:10000]) for pth, txt in get_db(datapath) if pth in _neighbors_pths2idx and (pth in MyDataset.train_set or pth in MyDataset.test_set)]
            pool = Pool(processes=4)
            documents_to_words = dict(tqdm(pool.imap_unordered(convert_text, complete_db, chunksize=1), total=len(complete_db), leave=False, desc='Convert to Fixed Dict'))
            pool.close()
            # Convert words to GT word vector
            for pth, sentence in tqdm(documents_to_words.items(), leave=False, desc='Words to Vals'):
                if not sentence:
                    continue
                sentence_to_idx = np.asarray([word_to_index['<!START!>']]+[word_to_index[word] for word in sentence]+[word_to_index['<!END!>']], dtype=np.int32)
                MyDataset.result_db[pth] = sentence_to_idx
            ##########################################################
        if mode == 'train':
            self.dataset = [img for img in MyDataset.train_set if img in MyDataset.result_db]
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        elif mode == 'val':
            self.dataset = [img for img in MyDataset.test_set if img in MyDataset.result_db]
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        while True:
            imgpath = os.path.join(self.base_path, self.dataset[index])
            print(f"trying to get img {imgpath}")
            img = Image.open(imgpath).convert('RGB')
            print(f"got img at {imgpath}")
            neighbor_data = get_doc2vec_neighbors(self.dataset[index], self.transform, img, self.base_path)
            print(f"got neighbor data for {imgpath}")
            try:
                imgpath = os.path.join(self.base_path, self.dataset[index])
                print(f"trying to get img {imgpath}")
                img = Image.open(imgpath).convert('RGB')
                print(f"got img at {imgpath}")
                neighbor_data = get_doc2vec_neighbors(self.dataset[index], self.transform, img, self.base_path)
                print(f"got neighbor data for {imgpath}")
                break
            except:
                index = random.randrange(len(self.dataset))
                continue        
        # img, path, neighbor_img, neighbor_pth        
        return self.transform(img), imgpath, neighbor_data[0], neighbor_data[1]
    

def get_doc2vec_neighbors(pth, transform, orig_img, base_path):
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
            img = transform(Image.open(os.path.join(base_path, neighbor_pth)).convert('RGB'))
        except:
            continue
        neighbor_imgs.append(img)
        neighbor_pths.append(neighbor_pth)
    # USING 1 NEIGHBOR
    return (neighbor_imgs[0], neighbor_pths[0])