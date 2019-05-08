import yaml
import model as cnn_model


#reading the model parameters from the yaml config file
config_dict = yaml.load(open('config.yaml'))

# name of the CNN model to be run
model = config_dict['model']

pretrained = config_dict['pretrained']
finetuning = config_dict['fine_tuning']
batch_size = config_dict['batch_size']

weight_decay = config_dict['weight_decay']
learning_rate = config_dict['learning_rate']

optmizer = config_dict['optimizer']

tune_epochs = config_dict['tune_epochs']
train_epochs = config_dict['train_epochs']

images_dir = config_dict['images_dir']
labels_dir = config_dict['labels_dir']

# pritning config details for logging purpose
print(config_dict)


# call the method to start training

# path to images # learning rate # weight_decay
cnn_model.train_model_handler(images_dir, learning_rate, weight_decay, config_dict)
