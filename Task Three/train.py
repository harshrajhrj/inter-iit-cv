import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random # generate random indexes to visualize some images randomly
import matplotlib.pyplot as plt

from vit import PatchEmbedding, MLP, TransformerEncoderLayer, VisionTransformer, HyperParameters

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")


# SET THE SEED
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)


# CREATING HYPER-PARAMETERS OBJECT
hp = HyperParameters()


# DEFINE IMAGE TRANSFORMATIONS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    # helps the model to converge faster and
    # also it helps to make numerical computations stable
])


# GETTING A DATASET
train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)


# CONVERTING DATASET INTO DATALOADER
train_loader = DataLoader(dataset=train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=hp.BATCH_SIZE, shuffle=False) # best practice not to shuffle


print(f"DataLoader: {train_loader, test_loader}")
print(f"Length of train_loader: {len(train_loader)} batches of {hp.BATCH_SIZE}....")
print(f"Length of test_loader: {len(test_loader)} batches of {hp.BATCH_SIZE}...")


# INSTANTIATE MODEL
model = VisionTransformer(
    hp.IMAGE_SIZE, hp.PATCH_SIZE, hp.CHANNELS, hp.NUM_CLASSES, hp.EMBED_DIM, hp.DEPTH, hp.NUM_HEADS, hp.MLP_DIM, hp.DROP_RATE
).to(device)


print(f"{model}")


# DEFINING A LOSS FUNCTION AND AN OPTIMIZER
criterion = nn.CrossEntropyLoss() # measure how wrong our model is
optimizer = optim.Adam(params=model.parameters(), lr=hp.LEARNING_RATE)


# DEFINING A TRAINING LOOP FUNCTION
def train(model, loader, optimizer, criterion):
    model.train()

    totalLoss, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # forward pass (model outputs raw logits)
        output = model(x)
        # calculate loss (per batch)
        loss = criterion(output, y)
        # perform backpropagation
        loss.backward()
        # perform gradient descent
        optimizer.step()
        totalLoss += loss.item() * x.size(0)
        correct += (output.argmax(1) == y).sum().item()

    # scale the loss (Normalization step to make the loss general across all batches)
    return totalLoss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader):
    model.eval() # set the mode of the model into evaluation
    correct = 0
    with torch.inference_mode():
        for x, y, in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            correct += (output.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


# TRAINING
from tqdm.auto import tqdm

train_accuracies = []
test_accuracies = []

for epoch in tqdm(range(hp.EPOCHS)):
    trainLoss, trainAccuracy = train(model, train_loader, optimizer, criterion)
    testAccuracy = evaluate(model, test_loader)
    train_accuracies.append(trainAccuracy)
    test_accuracies.append(testAccuracy)
    print(f"Epoch: {epoch+1}/{hp.EPOCHS}, Train loss: {trainLoss:.4f}, Train accuracy: {trainAccuracy:.4f}, Test accuracy: {testAccuracy:.4f}")


# PLOT ACCURACIES
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Test Accuracy")
plt.show()


# VISUALIZE PREDICTIONS
def predictAndPlotGrid(model, dataset, classes, gridSize=3):
    model.eval()
    fix, axes = plt.subplots(gridSize, gridSize, figsize=(9, 9))
    for i in range(gridSize):
        for j in range(gridSize):
            idx = random.randint(0, len(dataset) - 1)
            img, true_label = dataset[idx]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
            img = img / 2 + 0.5 # unnormalize images to be able to plot them with matplotlib
            npimg = img.cpu().numpy()
            axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)))
            truth = classes[true_label] == classes[predicted.item()]
            if truth:
                color = "g"
            else:
                color = "r"
            axes[i, j].set_title(f"Truth: {classes[true_label]}\n, Predicted: {classes[predicted.item()]}", fontsize=10, c=color)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()


predictAndPlotGrid(model, test_dataset, classes=train_dataset.classes, gridSize=3)