import torch
import pandas as pd
import ImageDataLoader as IDL
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np


def make_predictions(data_transforms, model, images_dir, config_dict):

    # fetching config params
    BATCH_SIZE = config_dict['batch_size']
    all_path_labels = config_dict['path_labels']
    num_workers = config_dict['num_workers']

    # use GPU if available
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)

    # set model to eval mode
    model.train(False)

    # create dataloader
    dataset = IDL.ImageDataSet(path_to_images=images_dir, fold="test", transform=data_transforms['val'], starter_images=True, sample=100)
    dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])


    ########################
    #   Model Eval         #
    ########################

    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]


            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if(i % 10 == 0):
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    for column in true_df:

        if column not in all_path_labels:
            continue

        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.as_matrix().astype(int), pred.as_matrix())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)

    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)

    return pred_df, auc_df
