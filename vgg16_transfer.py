
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

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(pretrained_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean = pretrained_means,
                         std = pretrained_stds)
])

test_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean = pretrained_means,
                         std = pretrained_stds)
])




ROOT = '.data'

train_data = datasets.CIFAR10(root=ROOT,
                              train=True,
                              download=True,
                              transform=train_transforms)

test_data = datasets.CIFAR10(root=ROOT,
                             train=False,
                             download=True,
                             transform=test_transforms)




VALID_RATIO = 0.9
n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data,
                                           [n_train_examples, n_valid_examples])



valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms




print('Number of training examples:', len(train_data))
print('Number of validation examples:', len(valid_data))
print('Number of test examples:', len(test_data))



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
  fig = plt.figure(figsize=(7,7))

  for i in range(n_images):
    ax = fig.add_subplot(rows, cols, i+1)
    image = images[i]
    if normalize:
      image = normalize_image(image)
    ax.imshow(image.permute(1,2,0).cpu().numpy())
    ax.set_title(classes[labels[i]])
    ax.axis('off')




N_IMAGES = 9

images, labels = zip(*[(image, label) for image, label 
                       in [train_data[i] for i in range(N_IMAGES)]])

classes = test_data.classes

plot_images(images, labels, classes)





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
        nn.Linear(4096, output_dim),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    h = x.view(x.shape[0], -1)
    output = self.classifier(h)
    return output, h



vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 
                512, 512, 'M', 512, 512, 'M']

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
                512, 512, 512, 'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
                512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']



def get_vgg_layers(config, batch_norm):
  layers = []
  in_channels = 3

  for c in config:
    assert c == 'M' or isinstance(c, int)
    if c == 'M':
      layers += [nn.MaxPool2d(2)]
    else:
      conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
      
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]

      in_channels = c

  return nn.Sequential(*layers)



vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)



vgg11_layers



OUTPUT_DIM = 10
model = VGG(vgg11_layers, OUTPUT_DIM)
model



pretrained_model = models.vgg11_bn(weights='IMAGENET1K_V1')
pretrained_model




# Replace the final layer of the pretrained model
IN_FEATURES = pretrained_model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pretrained_model.classifier[-1] = final_fc



pretrained_model.classifier

model.load_state_dict(pretrained_model.state_dict())



def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
             
print(f'The model has {count_parameters(model):,} trainable parameters')



BATCH_SIZE = 128

train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size = BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size = BATCH_SIZE)



START_LR = 1e-7

optimizer = optim.Adam(model.parameters(), lr=START_LR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)


# In[23]:


class LRFinder:
  def __init__(self, model, optimizer, criterion, device):
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.device = device
    torch.save(model.state_dict(), 'init_params.pt')

  def _train_batch(self, iterator):
    self.model.train()
    self.optimizer.zero_grad()
    x, y = iterator.get_batch()
    x = x.to(self.device)
    y = y.to(self.device)
    y_pred, _ = model(x)
    loss = self.criterion(y_pred, y)
    loss.backward()
    self.optimizer.step()
    return loss.item()

  def range_test(self, iterator, end_lr=10, num_iter=100,
                 smooth_f=0.05, diverge_th=5):
    lrs=[]
    losses=[]
    best_loss = float('inf')

    lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
    iterator = IteratorWrapper(iterator)

    for iteration in range(num_iter):
      loss = self._train_batch(iterator)
      lrs.append(lr_scheduler.get_last_lr()[0])
      
      # update lr
      lr_scheduler.step()

      if iteration > 0:
        loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
      
      if loss < best_loss:
        best_loss = loss
      
      losses.append(loss)

      if loss > diverge_th * best_loss:
        print("Stopping early, the loss has diverged")
        break

    # reset model to initial parameters
    model.load_state_dict(torch.load('init_params.pt'))
    return lrs, losses

class ExponentialLR(_LRScheduler):
  def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
    self.end_lr = end_lr
    self.num_iter = num_iter
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    cur_iter = self.last_epoch
    r = cur_iter / self.num_iter
    return [base_lr * ((self.end_lr / base_lr) ** r) for base_lr in self.base_lrs]

class IteratorWrapper():
  def __init__(self, iterator):
    self.iterator = iterator
    self._iterator = iter(iterator)

  def __next__(self):
    try:
      inputs, labels = next(self._iterator)
    except StopIteration:
      self._iterator = iter(self.iterator)
      inputs, labels, *_ = next(self._iterator)
    return inputs, labels

  def get_batch(self):
    return next(self)



END_LR = 10
NUM_ITER = 100

lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)


def plot_lr_finder(lrs, losses, skip_start=5, skip_end=5):
  if skip_end == 0:
    lrs = lrs[skip_start:]
    losses = losses[skip_start:]
  else:
    lrs = lrs[skip_start:-skip_end]
    losses = losses[skip_start:-skip_end]

  fig = plt.figure(figsize=(16,8))
  ax = fig.add_subplot(1,1,1)
  ax.plot(lrs, losses)
  ax.set_xscale('log')
  ax.set_xlabel('Learning rate')
  ax.set_ylabel('Loss')
  ax.grid(True, 'both', 'x')
  plt.show()




plot_lr_finder(lrs, losses, skip_start=10, skip_end=20)


FOUND_LR = 1e-4

params = [
    {'params': model.features.parameters(), 'lr': FOUND_LR / 10},
    {'params': model.classifier.parameters()}
]

optimizer = optim.Adam(params, lr=FOUND_LR)


def calculate_accuracy(y_pred, y):
  top_pred = y_pred.argmax(axis=1, keepdims=True)
  correct = top_pred.eq(y.view_as(top_pred)).sum()
  acc = correct.float() / y.shape[0]
  return acc


def train(model, iterator, optimizer, criterion, device):
  epoch_loss = 0
  epoch_acc = 0
  model.train()
  for (x,y) in tqdm(iterator, desc='Training', leave=False):
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    y_pred, _ = model(x)
    loss = criterion(y_pred, y)
    acc = calculate_accuracy(y_pred, y)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()
  return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion, device):
  epoch_loss = 0
  epoch_acc = 0
  model.eval()

  with torch.no_grad():
    for (x, y) in tqdm(iterator, desc='Evaluating', leave=False):
      x = x.to(device)
      y = y.to(device)
      y_pred, _ = model(x)
      loss = criterion(y_pred, y)
      acc = calculate_accuracy(y_pred, y)
      epoch_loss += loss.item()
      epoch_acc += acc.item()

  return epoch_loss / len(iterator), epoch_acc / len(iterator)



def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_min = int(elapsed_time/60)
  elapsed_secs = int(elapsed_time - elapsed_min * 60)
  return elapsed_min, elapsed_secs



EPOCHS=6
best_valid_loss = float('inf')

for epoch in trange(EPOCHS):
  start_time = time.monotonic()

  train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
  valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'vgg-transfer-model.pt')

  end_time = time.monotonic()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
  print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')


model.load_state_dict(torch.load('vgg-transfer-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
print(f'Test loss: {test_loss:.3f} | Test accuracy: {test_acc*100:.2f}%')



def plot_image(image):
  """
  image -- tensor of shape (C,H,W)
  """
  image = copy.deepcopy(image)
  image = normalize_image(image)
  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(1,1,1)
  ax.imshow(image.permute(1,2,0).cpu().numpy())
  ax.axis('off')



def predict(image):
  """
  image -- tensor of shape (C,H,W)
  """
  image = copy.deepcopy(image)

  model.eval()
  with torch.no_grad():
    image = image.unsqueeze(0)
    image = image.to(device)
    pred_label, _ = model(image)
    pred_probs = F.softmax(pred_label, dim=-1)
    pred_value, pred_id = torch.max(pred_probs, dim=-1)
  return pred_value, pred_id


idx = 14

image, label = test_data[idx]

plot_image(image)

pred_value, pred_id = predict(image)

print(f'True label: {classes[label]}')
print(f'Pred. label: {classes[pred_id.item()]} ({pred_value.item()*100:.3f}%)')



def get_predictions(model, iterator, device):
  model.eval()
  images = []
  labels = []
  probs = []

  with torch.no_grad():
    for (x,y) in iterator:
      x = x.to(device)
      y_pred, _ = model(x)
      y_prob = F.softmax(y_pred, dim=-1)
      images.append(x.cpu())
      labels.append(y.cpu())
      probs.append(y_prob.cpu())

  images = torch.cat(images, dim=0)
  labels = torch.cat(labels, dim=0)
  probs = torch.cat(probs, dim=0)
  
  return images, labels, probs


images, labels, probs = get_predictions(model, test_iterator, device)
pred_labels = torch.argmax(probs, dim=1)


def plot_confusion_matrix(labels, pred_labels, classes):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1,1,1)
  cm = confusion_matrix(labels, pred_labels)
  cm = ConfusionMatrixDisplay(cm, display_labels=classes)
  cm.plot(values_format='d', cmap='Blues', ax=ax)
  plt.xticks(rotation=20)




plot_confusion_matrix(labels, pred_labels, classes)





corrects = torch.eq(labels, pred_labels)
incorrect_examples = []

for image, label, prob, correct in zip(images, labels, probs, corrects):
  if not correct:
    incorrect_examples.append((image, label, prob))

incorrect_examples.sort(reverse=True,
                        key = lambda x : torch.max(x[2], dim=0).values) # x[2] is prob



def plot_most_incorrect(incorrect, classes, n_images, normalize=True):
  rows = int(np.sqrt(n_images))
  cols = int(np.sqrt(n_images))
  fig = plt.figure(figsize=(25,20))

  for i in range(n_images):
    ax = fig.add_subplot(rows, cols, i+1)
    image, true_label, probs = incorrect[i]
    image = image.permute(1,2,0)
    true_prob = probs[true_label]
    incorrect_prob, incorrect_label = torch.max(probs, dim=0) 
    # needs 'dim=0' to return both value and index
    true_class = classes[true_label]
    incorrect_class = classes[incorrect_label]

    if normalize:
      normalize_image(image)

    ax.imshow(image.cpu().numpy())
    ax.set_title(f'True label: {true_class} ({true_prob:.3f})\n'
                f'Pred. label: {incorrect_class} ({incorrect_prob:.3f})')
    ax.axis('off')

  fig.subplots_adjust(hspace=0.4)




N_IMAGES = 36

plot_most_incorrect(incorrect_examples, classes, N_IMAGES)





