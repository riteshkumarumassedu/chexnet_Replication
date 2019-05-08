import os
import time
from shutil import rmtree
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms

import ImageDataLoader as IDL
import model_predict as pred

from __future__ import print_function, division

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))

####################################
#           TO save model state    #
####################################

def checkpoint(model, best_loss, epoch, LR):

    state = { 'model': model, 'best_loss': best_loss, 'epoch': epoch, 'rng_state': torch.get_rng_state(),'LR': LR }

    torch.save(state, 'results/checkpoint')


###########################################
#           Handles the model training    #
###########################################

def training_helper(model, criterion, optimizer, LR, num_epochs, dataloaders, dataset_sizes, weight_decay, config_dict):

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)


    since = time.time()
    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1


    ####################################
    #           Training segment       #
    ####################################

    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0

            for data in dataloaders[phase]:

                # parse the data and load to device GPU/CPU
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device)).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decreasing LR  from " + str(LR) + " to " +  str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10

                # create new optimizer with lower learning rate
                optimizer = optim.SGD( filter( lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=config_dict['momentum'], weight_decay=weight_decay)
                print("Updated  LR :  " + str(LR))


            # saving the model if it has got best loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 5 epochs
        if ((epoch - best_epoch) >= 5):
            print("Stopping training since no imrovement over past 5 epochs")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch

##############################
#       Train Model Handler  #
##############################

def train_model_handler(PATH_TO_IMAGES, LR, WEIGHT_DECAY, config_dict):

    # fetching config params

    NUM_EPOCHS = config_dict['train_epochs']

    BATCH_SIZE = config_dict['batch_size']

    mean = config_dict['image_mean']

    std  = config_dict['image_std']

    N_LABELS = config_dict['class_labels']

    momentum = config_dict['momentum']

    # use GPU if available

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)


    ####################################
    #           Data Loader part       #
    ####################################

    try:
        rmtree('results/')
    except BaseException:
        pass
    os.makedirs("results/")


    df = pd.read_csv( config_dict['nih_labels_csv'], index_col=0)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


    transformed_datasets = {}
    transformed_datasets['train'] = IDL.ImageDataSet(path_to_images=PATH_TO_IMAGES, fold='train', transform=data_transforms['train'], starter_images=True, sample=100)
    transformed_datasets['val'] = IDL.ImageDataSet(path_to_images=PATH_TO_IMAGES, fold='val', transform=data_transforms['val'], starter_images=True, sample=100)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(transformed_datasets['train'],batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader( transformed_datasets['val'],batch_size=BATCH_SIZE,shuffle=True,num_workers=8)


    ####################################
    #   Diff. CNN models support       #
    ####################################

    model_from_config = config_dict['model']

    if model_from_config =='densenet121':
        model = models.densenet121(pretrained=config_dict['pretrained'])
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_features, N_LABELS), nn.Sigmoid())
        model = model.to(device)

    if model_from_config == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=config_dict['pretrained'])
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(nn.Linear(num_features, N_LABELS), nn.Sigmoid())
        model = model.to(device)

    if model_from_config == 'resnet34':
        model = models.resnet34(pretrained=config_dict['pretrained'])
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, N_LABELS), nn.Sigmoid())
        model = model.to(device)

    if model_from_config == 'resnet50':
        model = models.resnet50(pretrained=config_dict['pretrained'])
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, N_LABELS), nn.Sigmoid())
        model = model.to(device)


    ####################################
    #           Model Optimizer        #
    ####################################
    criterion = nn.BCELoss()

    optimizer = optim.SGD(filter( lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=float(momentum), weight_decay=float(WEIGHT_DECAY))
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}


    ####################################
    #           Model Training         #
    ####################################
    model, best_epoch = training_helper(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                        dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY, config_dict=config_dict)

    ####################################
    #           Model Test             #
    ####################################
    preds, aucs = pred.make_predictions(data_transforms, model, PATH_TO_IMAGES, config_dict=config_dict)

    return preds, aucs
