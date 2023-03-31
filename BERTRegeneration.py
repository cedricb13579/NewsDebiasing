# verify GPU availability
#import tensorflow as tf

#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))

import torch
print("Using GPU: ", torch.cuda.is_available())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForMaskedLM
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from torchtext.datasets import WikiText2

def testOnData(sentences, bias_distribution, threshold):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  tokenized_texts = [tokenizer.tokenize(sample) for sample in sentences]
  print(tokenized_texts)
  for j in range(len(bias_distribution)):
    for i in range(len(bias_distribution[j])):
      if (bias_distribution[j][i] > threshold):
        print(i)
        print(j)
        print()
        tokenized_texts[j][i] = '?'

  print(tokenized_texts)
  exit(0)


train_iter = WikiText2(split='train')

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

test_sentence = "[CLS] Mitt Romney [MASK] on the senate floor [SEP]"
#test_bias_distribution = [[0, 0, 0.9, 0], [0, 0, 0, 0, 0, 0, 0, 0.9, 0], [0, 0, 0, 0.85, 0, 0, 0, 0, 0]]
#testOnData(test_sentences)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#Tokensize wikidataset
tokenized_wiki = [tokenizer.tokenize(sample) for sample in train_iter]

#Tokenize and segment the test sample
test_tokens = tokenizer.tokenize(test_sentence)
indexed_test = tokenizer.convert_tokens_to_ids(test_tokens)

#CLS = 101, SEP = 102, MASK = 103
segment_ids = [0 for i in range(len(test_tokens))]

tokens_tensor = torch.tensor([indexed_test])
segments_tensors = torch.tensor([segment_ids])

print ("Tokenize the 100th sentence:")
print (tokenized_wiki[100])
print(test_tokens)

# Set the maximum sequence length. 
MAX_LEN = 512
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_wiki],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_wiki]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

labels = torch.tensor([x for x in input_ids])
rand = torch.rand(input_ids.shape)

mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102) * (input_ids != 103) * (input_ids != 0)
for i in range(input_ids.shape[0]):
  selection = torch.flatten(mask_arr[i].nonzero()).tolist()
  input_ids[i, selection] = 103  # [MASK] token == 103

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

print(input_ids.shape)


# Use train_test_split to split our data into train and validation sets for training
train_inputs, validation_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=2023, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2023, test_size=0.1)
                                             
# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. 
batch_size = 32

# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")


# BERT fine-tuning parameters
optimizer = BertAdam(model.parameters(), lr=2e-5)

train_loss_set = []
epochs = 2
for e in range(epochs):
  print(epochs)
  model.train()
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  for step, batch in enumerate(train_dataloader):

    b_input_ids, b_input_mask, b_labels = batch
    optimizer.zero_grad()

    loss = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask.long(), masked_lm_labels=b_labels.long())

    train_loss_set.append(loss.item())

    loss.backward()
    optimizer.step()

    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1
  print("Train loss: {}".format(tr_loss/nb_tr_steps))


tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

model.eval()

with torch.no_grad():
  predictions = model(tokens_tensor, segments_tensors)

masked_index = 4
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
print(predicted_token)