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
import pandas as pd
import time
from PIL import Image
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
# Dataset Normalization                                                  #
##########################################################################

def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224,224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out

##########################################################################
# Dataset Handler                                                        #
##########################################################################

def test_data_from_fname(fname):
    im = Image.open(f'{test_dir}/{fname}')
    return apply_test_transforms(im)

##########################################################################
# Load The Model                                                         #
##########################################################################

# Download the pre-trained model of ResNet-50
model_conv  = torchvision.models.resnet50(pretrained=True)

# Parameters of newly constructed modules have requires_grad=True by default
for param in model_conv.parameters():
    param.requires_grad = False

# We change the parameter of the final fully connected layer.
# We have to keep the number of input features to this layer.
# We change the output features from this layer into 2 features (i.e., we only have two classes).
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

# Define the checkpoint location to save the trained model
check_point = f'{chk_dir}/model-checkpoint-fp32.tar'

# Load the check_point
checkpoint = torch.load(check_point)
print("Checkpoint Loaded")
print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
model_conv.load_state_dict(checkpoint['model_state_dict'])
 
# Copy the model to GPU memory
model_conv = model_conv.to(device)

# Set the model to eval mode
model_conv.eval()

##########################################################################
# Define Prediction Function                                             #
##########################################################################

def predict_dog_prob_of_single_instance(model, tensor):
    with torch.autograd.profiler.emit_nvtx():
        batch = torch.stack([tensor])
        batch = batch.to(device) # Send the input to GPU
        softMax = nn.Softmax(dim = 1)
        preds = softMax(model(batch))
    return preds[0,1].item()
    
##########################################################################
# Prediction                                                             #
##########################################################################

###################### Change as needed ######################
num_of_test_images = 200
##############################################################

test_data_files = os.listdir(test_dir)

if(num_of_test_images<2) :
    num_of_test_images = 2

df = pd.DataFrame(columns = ['ImageName', 'Predicted', 'Probability'])
image_inferenced = 0

since = time.time()
profiler.start()
for fname in test_data_files :    
    im         = Image.open(f'{test_dir}/{fname}')
    imstar     = apply_test_transforms(im)    
    outputs    = predict_dog_prob_of_single_instance(model_conv, imstar)

    if(outputs<0.5) :
        df.loc[image_inferenced] = [fname, 'cat', str(1-outputs)]
    else :
        df.loc[image_inferenced] = [fname, 'dog', str(outputs)]
        
    if image_inferenced % 100 == 0:
        print(' Finished predicting image %d' % (image_inferenced))
        
    image_inferenced += 1
    if(image_inferenced>=num_of_test_images) :
        break
        
profiler.stop()
time_elapsed = time.time() - since

print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

df.to_csv('infer_fp32_result.csv', index = False)