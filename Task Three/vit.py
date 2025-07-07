import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader
import torchvision
# from torchvision import datasets, transforms
# import random
# import numpy as np
# import matplotlib.pyplot as plt

print(f"{torch.__version__}")
print(f"{torchvision.__version__}")



# PATCH EMBEDDING
class PatchEmbedding(nn.Module): # we need to split our images into patches
    def __init__(self, imageSize, patchSize, inChannels, embedDim):
        super().__init__()
        self.patchSize = patchSize
        # each patch is flattened into a vector and passed through a **linear projection layer** to embed it into a 
        # fixed-length vector (like word embeddings in NLP)
        self.projection = nn.Conv2d(in_channels=inChannels, out_channels=embedDim, kernel_size=patchSize, stride=patchSize)
        numPatches = (imageSize // patchSize) ** 2 # integer division
        self.clsToken = nn.Parameter(torch.randn(1, 1, embedDim)) # initialize weights randomly
        self.posEmbedding = nn.Parameter(torch.randn(1, 1+numPatches, embedDim)) # represents positional information

    def forward(self, x : torch.Tensor):
        B = x.size(0) # to receive the size for X at index '0'
        x = self.projection(x) # (B, E, H/P, w/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, E) to make sure shapes align
        clsToken = self.clsToken.expand(B, -1, -1)
        x = torch.cat((clsToken, x), dim=1)
        x = x + self.posEmbedding
        return x


# FEED FORWARD NN
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropRate):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features)
        self.dropout = nn.Dropout(dropRate)

    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
    

# TRANSFORMER ENCODER LAYER FOLLOWING THE ARCHITECTURE
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedDim, numHeads, mlpDim, dropRate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedDim)
        self.attention = nn.MultiheadAttention(embed_dim=embedDim, num_heads=numHeads, dropout=dropRate, batch_first=True)
        self.norm2 = nn.LayerNorm(embedDim)
        self.mlp = MLP(embedDim, mlpDim, dropRate)

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    

# VISION TRANSFORMER
class VisionTransformer(nn.Module):
    def __init__(self, imageSize, patchSize, in_channels, numClasses, embedDim, depth, numHeads, mlpDim, dropRate):
        super().__init__()
        self.patchEmbed = PatchEmbedding(imageSize=imageSize, patchSize=patchSize, inChannels=in_channels, embedDim=embedDim)
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embedDim, numHeads, mlpDim, dropRate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedDim)
        self.head = nn.Linear(embedDim, numClasses)

    def forward(self, x):
        x = self.patchEmbed(x)
        x = self.encoder(x)
        x = self.norm(x)
        clsToken = x[:, 0] # doing indexing
        return self.head(clsToken) # output 10 logits


# SETTING THE HYPER-PARAMETERS
class HyperParameters():
    def __init__(self):
        self.BATCH_SIZE = 128
        self.EPOCHS = 10 # trying increasing the number of epochs to 30
        self.LEARNING_RATE = 3e-4
        self.PATCH_SIZE = 4
        self.NUM_CLASSES = 10
        self.IMAGE_SIZE = 32 # transform the image and make the size go to 224
        self.CHANNELS = 3
        self.EMBED_DIM = 256
        self.NUM_HEADS = 8 # increase the number of heads
        self.DEPTH = 6
        self.MLP_DIM = 512
        self.DROP_RATE = 0.1
