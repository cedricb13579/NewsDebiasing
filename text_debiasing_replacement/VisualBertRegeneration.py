import torch
#import tensorflow as tf
print("Using GPU: ", torch.cuda.is_available())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, VisualBertForPreTraining
from pytorch_pretrained_bert import BertAdam
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from utils import Config
from visualizing_image import SingleImageViz
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from random import randrange
import ast
import imageio
import gensim.models
import sys

torch.cuda.empty_cache()
def train(num_samples):
    print("Loading Models")
    wikiImageData = pd.read_csv('wikiImageText/wit_v1.train.all-1percent_sample.tsv', sep='\t')
    imageUrl = wikiImageData["image_url"][0:num_samples]
    captionAttribute = wikiImageData["caption_attribution_description"][0:num_samples]
    captionAttribute = captionAttribute.astype(str).values.tolist()
    imageUrl = imageUrl.astype(str).values.tolist()

    # load models and model components
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)

    model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    #Freeze all but the last 4 model layers
    for name, param in list(model.named_parameters())[:-4]:
      param.requires_grad = False

    for name, param in list(model.named_parameters())[-4:]:
        print(name)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #Prepare image features
    images, sizes, scales_yx = image_preprocess(imageUrl)

    #Run frcnn
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    image_features = output_dict.get("roi_features")
    print(image_features.dtype)

    #Prepare masked tokens
    maskedCaptionAttributes = []
    for i in range(len(captionAttribute)):
    	sentence = captionAttribute[i].split()
    	toMask = randrange(len(sentence))
    	sentence[toMask] = "[MASK]"
    	maskedCaptionAttributes.append(" ".join(sentence))

    #Tokenize labels and ground truths
    MAX_LEN=512
    tokenized_captions = [tokenizer.tokenize(sample) for sample in maskedCaptionAttributes]
    tokenized_true_captions = [tokenizer.tokenize(sample) for sample in captionAttribute]


    inputs = tokenizer(
        maskedCaptionAttributes,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    labels = tokenizer(
        captionAttribute,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    tokenized_captions = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_captions],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    tokenized_captions = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_captions]

    tokenized_true_captions = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_true_captions]
    tokenized_true_captions = pad_sequences(tokenized_true_captions, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    #Set batches and prepare trainloader
    train_input_ids, validation_input_ids, train_label_ids, validation_label_ids = train_test_split(inputs.input_ids, labels.input_ids, random_state=2023, test_size=0.1)
    train_input_attention, validation_input_attention, train_label_attention, validation_label_attention = train_test_split(inputs.attention_mask, labels.attention_mask, random_state=2023, test_size=0.1)
    train_input_type, validation_input_type, train_label_type, validation_label_type = train_test_split(inputs.token_type_ids, labels.token_type_ids, random_state=2023, test_size=0.1)
    train_images, validation_images = train_test_split(image_features, test_size=0.1, random_state=2023)

    # Convert all of our data into torch tensors
    #train_inputs = torch.tensor(train_inputs)
    #validation_inputs = torch.tensor(validation_inputs)
    #train_labels = torch.tensor(train_labels)
    #validation_labels = torch.tensor(validation_labels)
    #train_images = torch.tensor(train_images)
    #validation_images = torch.tensor(validation_images)

    batch_size = 2
    train_data = TensorDataset(train_input_ids, train_input_attention, train_input_type, train_images, train_label_ids, train_label_attention, train_label_type)
    validation_data = TensorDataset(validation_input_ids, validation_input_attention, validation_input_type, validation_images, validation_label_ids, validation_label_attention, validation_label_type)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # BERT fine-tuning parameters
    optimizer = BertAdam(model.parameters(), lr=2e-5)

    train_loss_set = []
    epochs = 2

    for e in range(epochs):
      model.train()
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      i = 0
      for step, batch in enumerate(train_dataloader):
        print(f"batch {i}/{int(len(train_dataloader)/batch_size)}")
        i = i + 1
        b_input_ids, b_input_attention, b_input_type, b_input_images, b_label_ids, b_label_attention, b_label_type = batch
        optimizer.zero_grad()

        augment = torch.zeros([len(b_label_ids), 36])

        total_labels = torch.cat((b_label_ids, augment), 1)
        total_labels = total_labels.long()
        print(f"Visual Embeddings Shape {b_input_images.shape}")
        output = model(
            input_ids=b_input_ids,
            attention_mask=b_input_attention,
            visual_embeds=b_input_images,
            visual_attention_mask=torch.ones(b_input_images.shape[:-1]),
            token_type_ids=b_input_type,
            output_attentions=False,
            labels=total_labels
        )
        loss = output.loss
        train_loss_set.append(loss.item())
        print(loss)
        loss.backward()
        optimizer.step()

        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        print(f"Batch loss: {loss.item()}")
        tr_loss = tr_loss + loss.item()
      print("Train loss: {}".format(tr_loss/nb_tr_steps))
    torch.save(model, "VisualBert.pt")

def example():
    model = torch.load("VisualBert.pt")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    original_lines = [
        "The first time the senator spoke on the floor",
        "Twelve migrants found crossing the border",
        "Donald Trump makes a stupid deal stupider"
    ]

    test_lines = [
    	"The first time the senator [MASK] on the floor",
    	"Twelve [MASK] found crossing the border",
    	"Donald Trump makes a [MASK] deal stupider"
    ]

    for test_question in test_lines:
        test_question = [test_question]

        inputs = tokenizer(
            test_question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        masked_index = inputs.input_ids.tolist()[0].index(103)


        features = torch.from_numpy(np.zeros((1,36,2048))).float()

        output = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features,
            visual_attention_mask=torch.ones(features.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )
        # get prediction
        print(output.prediction_logits.shape)
        predicted_index = torch.argmax(output.prediction_logits[0, masked_index]).item()
        print(predicted_index)
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        print(predicted_token)

def test(num_samples):
    print("Starting test")

    #Load Models
    model = torch.load("VisualBert.pt")
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    vector_model = "./wordVecModel/wiki-news-300d-1M-subword.vec"
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(vector_model, binary=False)

    #Load CSV data
    bias_data = pd.read_csv("BiasedWordIdentificationTableRun1.csv").to_numpy()
    #0 == id
    #1 == id
    #2 == text
    #3 == array of tuples

    i = 0
    index = 0
    avg_cosine_loss = 0
    while i < num_samples:
        #replace most biased word with MASK
        line = ast.literal_eval(bias_data[index][3])
        if line == []:
            index = index + 1
            continue

        biased_word = line[0][0]
        punctuation = ".,?!'~@#$%^&*()[]{}"
        j = 1
        while (biased_word in punctuation):
            biased_word = line[j][0]
            j = j + 1

        input_text = bias_data[index][2]
        input_text = input_text.replace(" " + biased_word + " ", " [MASK] ", 1)
        
        #tokenize sentence
        inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        try:
            masked_index = inputs.input_ids.tolist()[0].index(103)
        except:
            index = index + 1
            continue

        #load image
        tag = str(bias_data[index][0])
        image_name = tag + ".jpeg"

        #process image through rcnn
        image = imageio.imread("./new_imgs/new_imgs/" + image_name)
        images, sizes, scales_yx = image_preprocess(image, single_image=True)
        images = images[None, :, :, :]
        sizes = sizes[None, :]
        scales_yx = scales_yx[None, :]

        #Run frcnn
        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        features = output_dict.get("roi_features")

        #Use BERT to make prediction
        #features = torch.from_numpy(np.zeros((1,36,2048))).float()

        output = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_embeds=features,
            visual_attention_mask=torch.ones(features.shape[:-1]),
            token_type_ids=inputs.token_type_ids,
            output_attentions=False,
        )

        # get prediction
        predicted_index = torch.argmax(output.prediction_logits[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        #print(predicted_token)

        #get cosine loss
        try:
            print(f"Sample {i}/{num_samples}")
            print(f"Original Word: {biased_word}")
            print(f"Predicted Word: {predicted_token}")
            distance = word_vectors.distance(biased_word, predicted_token[0])
            print(f"Cosine Similarity: {distance}")
        except:
            index = index + 1
            continue

        #increment
        i = i + 1
        index = index + 1
        avg_cosine_loss = avg_cosine_loss + distance
    avg_cosine_loss = avg_cosine_loss/num_samples
    print(f"Average Cosine Loss: {avg_cosine_loss}")
    return avg_cosine_loss

if __name__=="__main__":
    if len(sys.argv) != 3:
        print(f"Please include 2 command line arguments, train/test numSamples")
    task = sys.argv[1]
    numSamples = int(sys.argv[2])
    if task == "train":
        train(numSamples)
    elif task == "test":
        test(numSamples)
    else:
        print("First argument should be train or test")


