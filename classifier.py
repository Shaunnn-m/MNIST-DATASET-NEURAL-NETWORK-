# Data preprocessing
import numpy as np
import torch
from PIL import Image
from matplotlib import transforms, pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Setup directory and download flags
DATA_DIR = '.'
download_dataset = False

# Transformation for the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# MNIST datasets
train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

# Data reshaping
x_train = train_mnist.data.view(-1, 28 * 28).float() / 255.0
y_train = train_mnist.targets
x_test = test_mnist.data.view(-1, 28 * 28).float() / 255.0
y_test = test_mnist.targets

# Split dataset into training and validation sets
validation_size = int(0.8 * len(train_mnist))
training_size = len(train_mnist) - validation_size
train_data, val_data = random_split(train_mnist, [training_size, validation_size])

# Setup DataLoader for batches
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)


# Neural network model definition
class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, activations):
        super(NeuralNetwork, self).__init__()
        # Create layers based on the provided sizes and activation functions
        layers = [nn.Flatten()]
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if activations[i] is not None:
                layers.append(activations[i]())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


layer_sizes = [784, 256, 128, 10]  # Input layer, two hidden layers, output layer
activations = [nn.ReLU, nn.ReLU, None]  # Activation functions for each hidden layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(layer_sizes, activations).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training the model
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation function

def validate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


# Model training and validation loop with history tracking
train_losses = []
val_losses = []
val_accuracies = []

epochs = 10
for epoch in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, val_accuracy = validate(val_loader, model, loss_fn, device)

    # Record the metrics for history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# Evaluating model performance
test_loss, test_accuracy = validate(test_loader, model, loss_fn, device)
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

def load_image(image_path, device):
    # Define the transform for the input image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image to 28x28
        transforms.Grayscale(),       # Convert the image to grayscale
        transforms.ToTensor(),        # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
    ])

    # Open the image file
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Transform the image and add a batch dimension
    return image.to(device)


def make_prediction(model, image_path, device):
    image = load_image(image_path, device)
    model.eval()
    with torch.no_grad():
        prediction = model(image).argmax(1).item()
    return prediction

# User interaction for custom predictions
print("Model training complete. Enter an image path to classify.")
while True:
    image_path = input("Enter an image path ('exit' to quit): ")
    if image_path == 'exit':
        break
    prediction = make_prediction(model, image_path, device)
    print(f"Predicted digit: {prediction}")