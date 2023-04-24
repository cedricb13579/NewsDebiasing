from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import gensim
import numpy as np
import pickle
import re, math
import os
import sys
import json
import random
import glob
import gzip
import time
import traceback
import torch
import json
from torch.autograd import Variable
from torchvision import transforms, models
from enum import Enum
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sklearn.utils
from collections import Counter
import csv, itertools
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from multiprocessing import Pool

from train_utils import get_db, convert_text
from train_utils import word_to_index, word_to_vector, _neighbors_pths2idx

from train_loss import angular_loss, bias_loss

from clip_utils import get_clip_components

from dataset import MyDataset

# TODO:
# -argparse instead of sys.argv
#     - 3 original inputs
#     -lr
#     -batch_size
#     -num_workers
# - replace img_model and rnn_model with clip encoders
# - finish new loss function

def main():
    vision_encoder, text_encoder, preprocess = get_clip_components()
    print("Getting train loader.")
    train_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('train', "E://Research//Political_Image_Debiasing//dataset//dataset_metadata_clean_5000.pickle"), batch_size=4, shuffle=True, num_workers=4)
    print("Getting val loader.")
    test_dataloader = torch.utils.data.DataLoader(dataset=MyDataset('val', "E://Research//Political_Image_Debiasing//dataset//dataset_metadata_clean_5000.pickle"), batch_size=4, shuffle=False, num_workers=4)
    print("Done loading datasets")
    writer = SummaryWriter(f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/')
    # img_model = torch.nn.DataParallel(ImageModel()).cuda()
    # rnn_model = torch.nn.DataParallel(RecurrentModel()).cuda()
    img_model = torch.nn.DataParallel(vision_encoder).cuda()
    rnn_model = torch.nn.DataParallel(text_encoder).cuda()
    img_model.train()
    rnn_model.train()
    optimizer = torch.optim.Adam(params=itertools.chain(img_model.parameters(), rnn_model.parameters()), lr=0.0001, weight_decay=1e-5)
    ### LR SCHEDULER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=5)
    ###
    itr = 0    
    best_loss = sys.maxsize
    for e in tqdm(range(1, 1000), ascii=True, desc='Epoch'):
        img_model.train()
        rnn_model.train()
        random.seed()
        with tqdm(total=len(train_dataloader), ascii=True, leave=False, desc='iter') as pbar:
            for i, (images, paths, neighbor_imgs, neighbor_pths) in enumerate(train_dataloader):
                itr += 1
                optimizer.zero_grad()
                images = images.float().cuda()
                print("after images to cuda")
                image_projections = img_model(images) # Batch size x 256
                print("after images projection")
                neighbor_imgs = neighbor_imgs.float().cuda()
                neighbor_imgs_projections = img_model(neighbor_imgs)
                print("after neighbors projection")

                neighbor_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in neighbor_pths]
                neighbor_lengths = torch.LongTensor([torch.numel(item) for item in neighbor_sentences])
                neighbor_sentences = torch.nn.utils.rnn.pad_sequence(sequences=neighbor_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()
                print("after neighbor sentences")
                positive_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in paths]
                pos_lengths = torch.LongTensor([torch.numel(item) for item in positive_sentences])
                positive_sentences = torch.nn.utils.rnn.pad_sequence(sequences=positive_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()
                print("after positive sentences")
                negative_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in random.sample(train_dataloader.dataset.dataset, len(positive_sentences))]
                neg_lengths = torch.LongTensor([torch.numel(item) for item in negative_sentences])
                negative_sentences = torch.nn.utils.rnn.pad_sequence(sequences=negative_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()
                print("after negative sentences")
                neighbor_projections = rnn_model(neighbor_sentences, neighbor_lengths)
                print("after neighbor sentence projections")
                positive_projections = rnn_model(positive_sentences, pos_lengths)
                print("after positive sentence projections")
                negative_projections = rnn_model(negative_sentences, neg_lengths)
                print("after negative sentence projections")

                # Baseline loss
                # Compute n-pairs angular loss
                image_projections_np = torch.repeat_interleave(image_projections, len(negative_projections), dim=0)
                positive_projections_np = torch.repeat_interleave(positive_projections, len(negative_projections), dim=0)
                negative_projections_np = negative_projections.repeat(len(negative_projections), 1)
                l_i2t = angular_loss(anchors=image_projections_np, positives=positive_projections_np, negatives=negative_projections_np)
                print("after n-pairs angular loss")

                # L_img - Image anchor, Neighbor img anchor, negative image neighbors. Angular Npairs
                image_projections_np = torch.repeat_interleave(image_projections, len(negative_projections), dim=0)
                neighbor_imgs_projections_np = torch.repeat_interleave(neighbor_imgs_projections, len(negative_projections), dim=0)
                permute_idxs = torch.from_numpy(np.asarray([j if i != j else (j+1) % len(image_projections) for i in range(len(image_projections)) for j in range(len(image_projections))]))
                image_projections_np2 = image_projections[permute_idxs,...]
                l_img = angular_loss(anchors=image_projections_np, positives=neighbor_imgs_projections_np, negatives=image_projections_np2)
                print("after image angular loss")

                # L_text - Angular npairs
                positive_projections_np = torch.repeat_interleave(positive_projections, len(negative_projections), dim=0)
                neighbor_projections_np = torch.repeat_interleave(neighbor_projections, len(negative_projections), dim=0)
                negative_projections_np = negative_projections.repeat(len(negative_projections), 1)
                l_text = angular_loss(anchors=positive_projections_np, positives=neighbor_projections_np, negatives=negative_projections_np)
                print("after text angular loss")

                # Symmetric angular loss npairs (text to image)
                positive_projections_np = torch.repeat_interleave(positive_projections, len(image_projections), dim=0)
                image_projections_np = torch.repeat_interleave(image_projections, len(image_projections), dim=0)                
                permute_idxs = torch.from_numpy(np.asarray([j if i != j else (j+1) % len(image_projections) for i in range(len(image_projections)) for j in range(len(image_projections))]))
                image_projections_np2 = image_projections[permute_idxs,...]
                l_sym = angular_loss(anchors=positive_projections_np, positives=image_projections_np, negatives=image_projections_np2)
                print("after adkjlsfhkajsjdhfjklhasdfj angular loss")

                # Bias loss scores
                

                loss = l_i2t + float(sys.argv[1])*l_sym + float(sys.argv[2])*l_img + float(sys.argv[3])*l_text

                loss.backward()
                optimizer.step()
                writer.add_scalar('data/train_loss', loss.item(), itr)
                writer.add_scalar('data/l_i2t', l_i2t.item(), itr)
                writer.add_scalar('data/l_sym', l_sym.item(), itr)
                writer.add_scalar('data/l_img', l_img.item(), itr)
                writer.add_scalar('data/l_text', l_text.item(), itr)

                pbar.update()
        img_model.eval()
        rnn_model.eval()
        losses = []
        random.seed(9485629)
        with tqdm(total=len(test_dataloader), ascii=True, leave=False, desc='eval') as pbar, torch.no_grad():
            for i, (images, paths, _, _) in enumerate(test_dataloader):
                optimizer.zero_grad()
                
                images = images.float().cuda()
                image_projections = img_model(images) # Batch size x 256

                positive_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in paths]
                pos_lengths = torch.LongTensor([torch.numel(item) for item in positive_sentences])
                positive_sentences = torch.nn.utils.rnn.pad_sequence(sequences=positive_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()

                negative_sentences = [torch.from_numpy(MyDataset.result_db[path]).long() for path in random.sample(test_dataloader.dataset.dataset, len(positive_sentences))]
                neg_lengths = torch.LongTensor([torch.numel(item) for item in negative_sentences])
                negative_sentences = torch.nn.utils.rnn.pad_sequence(sequences=negative_sentences, batch_first=True, padding_value=word_to_index['<!END!>']).cuda()

                positive_projections = rnn_model(positive_sentences, pos_lengths)
                negative_projections = rnn_model(negative_sentences, neg_lengths)

                loss = angular_loss(anchors=image_projections, positives=positive_projections, negatives=negative_projections)
                
                losses.append(loss.item())

                pbar.update()
        curr_loss = np.mean(losses)
        writer.add_scalar('data/val_loss', curr_loss, e)
        scheduler.step(curr_loss)
        # save only the best model
        if curr_loss < best_loss:
            best_loss = curr_loss
            # delete prior models
            prior_models = glob.glob(f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/*.pth')
            for pm in prior_models:
                os.remove(pm)
            try:
                torch.save(rnn_model.state_dict(), f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/rnn_model_{e}.pth')
                torch.save(img_model.state_dict(), f'models/l_i2t_l_sym_{sys.argv[1].replace(".","_")}_l_img_{sys.argv[2].replace(".","_")}_l_text_{sys.argv[3].replace(".","_")}/img_model_{e}.pth')
            except:
                print('Failed saving')
                continue
if __name__ == '__main__':
    main()