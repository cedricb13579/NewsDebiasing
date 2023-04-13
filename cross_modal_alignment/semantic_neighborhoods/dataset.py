import pickle
import random
import torch
import numpy as np

from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from torchvision import transforms

from train_utils import get_db, convert_text, get_doc2vec_neighbors
from train_utils import _neighbors_pths2idx, word_to_index

class MyDataset(torch.utils.data.Dataset):
    result_db = {}
    def __init__(self, mode):
        super().__init__()
        if not MyDataset.result_db:
            train_test_dict = pickle.load(open('train_test_pths.pickle', 'rb'))
            train_set = train_test_dict['train']
            val_set = set(random.sample(train_set, len(train_test_dict['test'])))
            train_set = [pth for pth in train_set if pth not in val_set]
            MyDataset.train_set = list(train_set)
            MyDataset.test_set = list(val_set)
            complete_db = [(pth, txt[:10000]) for pth, txt in get_db() if pth in _neighbors_pths2idx and (pth in MyDataset.train_set or pth in MyDataset.test_set)]
            pool = Pool(processes=48)
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
            try:
                img = Image.open(self.dataset[index]).convert('RGB')
                neighbor_data = get_doc2vec_neighbors(self.dataset[index], self.transform, img)
                break
            except:
                index = random.randrange(len(self.dataset))
                continue        
        # img, path, neighbor_img, neighbor_pth        
        return self.transform(img), self.dataset[index], neighbor_data[0], neighbor_data[1]