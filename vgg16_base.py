
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models

from sklearn import decomposition, manifold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

ROOT = '.data'
train_data = datasets.CIFAR10(root=ROOT, train=True, download=True)

means = train_data.data.mean(axis=(0, 1, 2)) / 255
stds = train_data.data.std(axis=(0, 1, 2)) / 255

print(f'Calculated means: {means}')
print(f'Calculated stds: {stds}')

train_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

train_data = datasets.CIFAR10(root=ROOT, train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root=ROOT, train=False, download=True, transform=test_transforms)

VALID_RATIO = 0.9
n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print('Number of training examples:', len(train_data))
print('Number of validation examples:', len(valid_data))
print('Number of test examples:', len(test_data))

print('Shape of an image: ', train_data[0][0].shape)


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, normalize=True):
    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize=(7, 7))

    for i in range(n_images):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]
        if normalize:
            image = normalize_image(image)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')


N_IMAGES = 9

images, labels = zip(*[(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]])

classes = test_data.classes

plot_images(images, labels, classes)


# Configure Model

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


def make_features(cfg: list, batch_norm: bool = False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


# VGG16 Configuration

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg16(output_dim):
    return VGG(make_features(cfg['VGG16']), output_dim)


# Hyperparameters

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(RANDOM_SEED)

train_loader = data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

valid_loader = data.DataLoader(dataset=valid_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)


# Train and Evaluate Functions

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, _ = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float() / num_examples * 100


def plot_results(train_losses, valid_losses, train_accs, valid_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(valid_accs, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Training Loop

def train(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_pred = 0
    for batch_idx, (features, targets) in enumerate(tqdm(data_loader)):
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits, _ = model(features)

        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += F.cross_entropy(logits, targets, reduction='sum').item()
        _, predicted_labels = torch.max(logits, 1)
        correct_pred += (predicted_labels == targets).sum()

    loss = running_loss / len(data_loader.dataset)
    accuracy = correct_pred.float() / len(data_loader.dataset) * 100
    return loss, accuracy


def evaluate(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct_pred = 0
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            logits, _ = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            running_loss += loss.item()
            _, predicted_labels = torch.max(logits, 1)
            correct_pred += (predicted_labels == targets).sum()

    loss = running_loss / len(data_loader.dataset)
    accuracy = correct_pred.float() / len(data_loader.dataset) * 100
    return loss, accuracy


# Main Function

def main():
    model = vgg16(output_dim=len(classes))
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, mode='max', verbose=True)

    best_accuracy = 0.0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):

        model.train()
        for batch_idx, (features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits, _ = model(features)

            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, train_loader, DEVICE)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, DEVICE)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), 'best_model_state.bin')

        scheduler.step(valid_accuracy)

        print(
            f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time

    print(f'\nTraining took {total_time / 60:.2f} minutes')

    model.load_state_dict(torch.load('best_model_state.bin'))

    test_loss, test_accuracy = evaluate(model, test_loader, DEVICE)
    print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')

    plot_results(train_losses, valid_losses, train_accs, valid_accs)

    # Generate Embeddings

    def get_embeddings(model, data_loader, device):
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(device)
                targets = targets.to(device)
                _, h = model(features)
                embeddings.append(h)
                labels.append(targets)
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels

    train_embeddings, train_labels = get_embeddings(model, train_loader, DEVICE)
    test_embeddings, test_labels = get_embeddings(model, test_loader, DEVICE)



    def visualize_embeddings(embeddings, labels, classes):
        pca = decomposition.PCA(n_components=2)
        pca.fit(embeddings.cpu())
        embeddings_pca = pca.transform(embeddings.cpu())
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels.cpu(), alpha=0.6,
                              cmap=plt.get_cmap('jet', 10))
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.show()

    visualize_embeddings(train_embeddings, train_labels, classes)
    visualize_embeddings(test_embeddings, test_labels, classes)


main()
