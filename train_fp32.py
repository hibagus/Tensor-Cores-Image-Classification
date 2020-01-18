##########################################################################
# Import Library                                                         #
##########################################################################

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import datetime
import time
import copy
import math
import torch.cuda.profiler as profiler
from apex import pyprof

pyprof.nvtx.init()

##########################################################################
# Global Variables                                                       #
##########################################################################

data_dir       = './'
raw_dir        = f'{data_dir}/raw'
raw_dogs_dir   = f'{raw_dir}/dogs'
raw_cats_dir   = f'{raw_dir}/cats'
train_dir      = f'{data_dir}/train'
train_dogs_dir = f'{train_dir}/dogs'
train_cats_dir = f'{train_dir}/cats'
val_dir        = f'{data_dir}/val'
val_dogs_dir   = f'{val_dir}/dogs'
val_cats_dir   = f'{val_dir}/cats'
log_dir        = f'{data_dir}/log'
chk_dir        = f'{data_dir}/checkpoint'
test_dir       = f'{data_dir}/test'

##########################################################################
# GPU Initialization                                                     #
##########################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########################################################################
# Dataset Augmentation and Normalization                                 #
##########################################################################

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

##########################################################################
# Dataset Handler                                                        #
##########################################################################

###################### Change as needed ######################
batch_size  = 1024
num_workers = 96
##############################################################

# Define the data transformation
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Define the data loader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
              for x in ['train', 'val']}

# Print the statistics
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
print(f'Train image size: {dataset_sizes["train"]}')
print(f'Validation image size: {dataset_sizes["val"]}')

##########################################################################
# Download Pre-Trained Model                                             #
##########################################################################

# Download the pre-trained model of ResNet-50
model_conv  = torchvision.models.resnet50(pretrained=True)

# Define the checkpoint location to save the trained model
check_point = f'{chk_dir}/model-checkpoint-fp32.tar'

##########################################################################
# Define Training Components                                             #
##########################################################################

# Parameters of newly constructed modules have requires_grad=True by default
for param in model_conv.parameters():
    param.requires_grad = False

# We change the parameter of the final fully connected layer.
# We have to keep the number of input features to this layer.
# We change the output features from this layer into 2 features (i.e., we only have two classes).
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

# Copy the model into GPU memory
model_conv = model_conv.to(device)

# Choose the Criterion as Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Optimize only the parameters of the final fully connected layer since we have changed them.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# This is our learning rate scheduler. Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

##########################################################################
# Define Training Function                                               #
##########################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs, timestamp):
    since = time.time()
    writer = SummaryWriter('log/'+timestamp+'-fp32')
    
    # Initialization
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_acc = 0.
    
    with torch.autograd.profiler.emit_nvtx():
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
        
                running_loss = 0.0
                running_corrects = 0
        
                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
        
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
        
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    
                    if phase == 'train' :
                        writer.add_scalar('Train/Current_Running_Loss', loss.item(), epoch*len(dataloaders[phase])+i)
                        writer.add_scalar('Train/Current_Running_Corrects', torch.sum(preds == labels.data), epoch*len(dataloaders[phase])+i)
                        writer.add_scalar('Train/Accum_Running_Loss', running_loss, epoch*len(dataloaders[phase])+i)
                        writer.add_scalar('Train/Accum_Running_Corrects', running_corrects, epoch*len(dataloaders[phase])+i)
                    else :
                        writer.add_scalar('Validation/Current_Running_Loss', loss.item(), epoch*len(dataloaders[phase])+i)
                        writer.add_scalar('Validation/Current_Running_Corrects', torch.sum(preds == labels.data), epoch*len(dataloaders[phase])+i)
                        writer.add_scalar('Validation/Running_Loss', epoch_loss, epoch*len(dataloaders[phase])+i)
                        writer.add_scalar('Validation/Running_Corrects', epoch_acc, epoch*len(dataloaders[phase])+i)
        
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                if phase == 'train' :
                    writer.add_scalar('Train/Loss', epoch_loss, epoch)
                    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                else :
                    writer.add_scalar('Validation/Loss', epoch_loss, epoch)
                    writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)
                
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print(f'New best model found!')
                    print(f'New record loss: {epoch_loss}, previous record loss: {best_loss}')
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        
            writer.flush()
            print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, best_loss, best_acc

##########################################################################
# Training The Model                                                     #
##########################################################################

###################### Change as needed ######################
num_epochs = 3
##############################################################

today = datetime.datetime.today() 
timestamp = today.strftime('%Y%m%d-%H%M%S')

# Start the training
profiler.start()
model_conv, best_val_loss, best_val_acc = train_model(model_conv,
                                                      criterion,
                                                      optimizer_conv,
                                                      exp_lr_scheduler,
                                                      num_epochs,
                                                      timestamp)
profiler.stop()

# Save the trained model for future use.
torch.save({'model_state_dict': model_conv.state_dict(),
            'optimizer_state_dict': optimizer_conv.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            'scheduler_state_dict' : exp_lr_scheduler.state_dict(),
            }, check_point)