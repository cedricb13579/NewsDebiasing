import torch
from torchvision import models
from train_utils import word_to_index, word_to_vector

def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        m.weight.data.uniform_()
        m.bias.data.zero_()
class ImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        # Projection from Image Features into join space
        self.projector = torch.nn.Linear(2048, 256, bias=True)
        self.projector.apply(weight_init)
    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.shape[0], -1) # flatten, preserving batch dim
        x = self.projector(x)
        # critical to normalize projections
        x = F.normalize(x, dim=1)
        return x        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=len(word_to_index), embedding_dim=200)
        # INITIALIZE THE EMBEDDING MODEL
        self.embedding.weight.data[:-2,:].copy_(torch.from_numpy(word_to_vector.vectors))
        self.GRU = torch.nn.GRU(input_size=200, hidden_size=512, batch_first=True)
        # Projection 
        self.projector = torch.nn.Linear(512, 256, bias=True)
        self.projector.apply(weight_init)
    def forward(self, input, lengths):        
        embedding = self.embedding(input) # embed the padded sequence
        # Use pack_padded_sequence to make sure the LSTM won’t see the padded items
        embedding = torch.nn.utils.rnn.pack_padded_sequence(input=embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        # run through recurrent model
        _, h_n = self.GRU(embedding)
        # flatten for linear
        h_n = h_n.contiguous()[0]
        projection = self.projector(h_n)
        # critical to normalize projections
        projection = F.normalize(projection, dim=1)
        return projection