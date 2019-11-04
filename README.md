# CheXnet paper replication

Provides Python code to reproduce model training, predictions, and heatmaps from the [CheXNet paper](https://arxiv.org/pdf/1711.05225) that predicted 14 common diagnoses using convolutional neural networks in over 100,000 NIH chest x-rays.


## NIH Dataset
To explore the full dataset, [download images from NIH (large, ~40gb compressed)](https://nihcc.app.box.com/v/ChestXray-NIHCC),
extract all `tar.gz` files to a single folder, and provide path as needed in code.

## Train your own model!
Please note: a GPU is required to train the model. You will encounter errors if you do not have a GPU available and CUDA installed and you attempt to retrain. With a GPU, you can retrain the model with `retrain.py`. Make sure you download the full NIH dataset before trying this. If you run out of GPU memory, reduce `BATCH_SIZE` from its default setting of 16.
