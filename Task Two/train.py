from seresnet import seresnet18
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Split the training dataset into training and validation sets
# train_size = int(0.8 * len(train_dataset))  # 80% for training
# val_size = len(train_dataset) - train_size  # 20% for validation
# train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create data loaders (to pass batch of 64 images at once)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Instantiate the model and move to device
model = seresnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1) # Learning rate scheduler
# early_stopping = EarlyStopping(patience=7, verbose=True) # Early stopping to prevent overfitting

# Training the model
def train():
    num_epochs = 20  # number of training epochs
    train_loss = {}
    validation_loss = {}
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # reset gradients
            outputs = model(images)  # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # backpropagation
            optimizer.step()  # update model weights

            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss[epoch] = loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

        # Evaluate on validation set
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0.0
        #     for images, labels in val_loader:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         output = model(images)
        #         loss = criterion(output, labels)
        #         val_loss += loss.item()
        #     print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss = {val_loss / len(val_loader):.4f}")

        # validation_loss[epoch] = val_loss / len(val_loader)

        # # Early stopping check
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break

        # model.train()  # set model back to training mode
        
        scheduler.step() # adjust learning rate

        print()
    
    return train_loss, validation_loss

train_loss, validation_loss = train()


# Plotting the training and validation loss
def plot():
    plt.figure(figsize=(10, 5))
    losses = [train_loss[i] for i in range(len(train_loss))]
    plt.plot(losses, label='Training Loss')
    # val_losses = [validation_loss[i] for i in range(len(validation_loss))]
    # plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

plot()


# Save the trained model
torch.save(model.state_dict(), "seresnet_model.pth")

# Load the model
# model = seresnet18(num_classes=10).to(device)
# model.load_state_dict(torch.load("seresnet_model.pth"))
# model.eval()  # Set to evaluation mode


# Testing the model
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")



# Prediction and visualization of test images
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.cpu().numpy()
    npimg = np.clip(npimg, 0, 1)  # Clip to valid range for imshow
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of test images
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Display images
imshow(torchvision.utils.make_grid(images))

# Get model predictions
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Print predictions
print("Predicted:", predicted.cpu().numpy())
print("Actual:   ", labels.cpu().numpy())

# Load the model
# model = seresnet18(num_classes=10).to(device)
# model.load_state_dict(torch.load("seresnet_model.pth"))
# model.eval()  # Set to evaluation mode

# load the best model
# model.load_state_dict(torch.load('checkpoint.pt'))