################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
from tqdm import tqdm
import nltk
import skimage.io as io

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):

        config_data = read_file_in_dir('./model_configs/', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']

        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        self.__criterion = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        prev_val_loss = float('inf')
        for epoch in tqdm(range(start_epoch, self.__epochs)):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            # early stop
            if val_loss > prev_val_loss:
                break
            else:
                prev_val_loss = val_loss
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
        

    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
            images = images.cuda()
            captions = captions.cuda()

            out = self.__model(images, captions)
            out, captions = out.reshape(-1, len(self.__vocab)), captions.reshape(-1)
            loss = self.__criterion(out, captions)
            training_loss += loss.item()

            loss.backward()
            self.__optimizer.step()
            self.__optimizer.zero_grad()
            

        return training_loss/(i+1)

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.cuda()
                captions = captions.cuda()

                out = self.__model(images, captions)
                out, captions = out.reshape(-1, len(self.__vocab)), captions.reshape(-1)
                loss = self.__criterion(out, captions)
                val_loss += loss.item()

        return val_loss/(i+1)

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu_1 = 0
        bleu_4 = 0

        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.cuda()
                captions = captions.cuda()

                out = self.__model(images, captions)
                out, captions = out.reshape(-1, len(self.__vocab)), captions.reshape(-1)
                loss = self.__criterion(out, captions)
                test_loss += loss.item()

                bleu_1_batch = 0
                bleu_4_batch = 0

                sampled_captions = self.__model.sample(images, self.__vocab, self.__generation_config['max_length'], self.__generation_config['deterministic'], self.__generation_config['temperature'])
                
                for i in range(len(img_ids)):
                    annotations = self.__coco_test.loadAnns(self.__coco_test.getAnnIds(img_ids[i]))
                    reference_captions = [a['caption'] for a in annotations]
                    reference_captions = [nltk.tokenize.word_tokenize(str(c).lower()) for c in reference_captions]

                    pred_captions = sampled_captions[i]
                    
                    bleu_1_batch += bleu1(reference_captions, pred_captions)
                    bleu_4_batch += bleu4(reference_captions, pred_captions)

                    
                bleu_1 += bleu_1_batch / (i+1)
                bleu_4 += bleu_4_batch / (i+1)
                
                # print(' '.join(pred_captions))
                # print('\n')


        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(test_loss/(iter+1),
                                                                                               np.exp(test_loss/(iter+1)),
                                                                                               bleu_1/(iter+1),
                                                                                               bleu_4/(iter+1))
        self.__log(result_str)

        return test_loss/(iter+1), bleu_1/(iter+1), bleu_4/(iter+1)

    def search_caption(self):
        self.__model.eval()

        if not os.path.exists('./captions/{}'.format(self.__name)):
            os.makedirs('./captions/{}'.format(self.__name))
            os.makedirs('./captions/{}/good_images'.format(self.__name))
            os.makedirs('./captions/{}/bad_images'.format(self.__name))

        good_count = 0
        bad_count = 0
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.cuda()
                captions = captions.cuda()

                sampled_captions = self.__model.sample(images, self.__vocab, self.__generation_config['max_length'],
                                                       deterministic=False,
                                                       temp=0.4)

                for i in range(len(img_ids)):
                    if good_count > 3 and bad_count > 3:
                        return 'finished finding captions!'
                    annotations = self.__coco_test.loadAnns(self.__coco_test.getAnnIds(img_ids[i]))
                    reference_captions = [a['caption'] for a in annotations]
                    reference_captions = [nltk.tokenize.word_tokenize(str(c).lower()) for c in reference_captions]

                    pred_captions = sampled_captions[i]

                    bleu_1_sample = bleu1(reference_captions, pred_captions)
                    bleu_4_sample = bleu4(reference_captions, pred_captions)

                    if bleu_1_sample > 90 and bleu_4_sample > 45 and good_count < 3:
                        good_count += 1
                        img = self.__coco_test.loadImgs(img_ids[i])
                        I = io.imread('%s/images/%s/%s' % ('./data', 'test', img[0]['file_name']))
                        plt.axis('off')
                        plt.imshow(I)
                        plt.savefig('./captions/{}/good_images/{}.png'.format(self.__name, img_ids[i]))
                        joined_reference_captions = []
                        for caption in reference_captions:
                            joined_reference_captions.append(' '.join(caption))

                        low_temperature_caption = self.__model.sample(images[i, :, :, :].unsqueeze(0), self.__vocab
                                                                      , self.__generation_config['max_length']
                                                                      , deterministic=False
                                                                      , temp=0.001)[0]
                        high_temperature_caption = self.__model.sample(images[i, :, :, :].unsqueeze(0), self.__vocab
                                                                       , self.__generation_config['max_length']
                                                                       , deterministic=False
                                                                       , temp=5)[0]
                        deterministic_caption = self.__model.sample(images[i, :, :, :].unsqueeze(0), self.__vocab
                                                                       , self.__generation_config['max_length']
                                                                       , deterministic=True
                                                                       , temp=0.4)[0]
                        with open('./captions/{}/good_images/{}_captions.txt'.format(self.__name, img_ids[i]), 'w') as f:
                            f.write('Actual captions:\n')
                            for caption in joined_reference_captions:
                                f.write(caption)
                                f.write('\n')
                            f.write('Predicted caption:\n')
                            f.write(' '.join(pred_captions))
                            f.write('\n')
                            f.write('Predicted caption with 0.001 temperature:\n')
                            f.write(' '.join(low_temperature_caption))
                            f.write('\n')
                            f.write('Predicted caption with 5 temperature:\n')
                            f.write(' '.join(high_temperature_caption))
                            f.write('\n')
                            f.write('Predicted caption with deterministic:\n')
                            f.write(' '.join(deterministic_caption))
                            f.write('\n')

                    if bleu_1_sample < 30 and bleu_4_sample < 3 and bad_count < 3:
                        bad_count += 1
                        img = self.__coco_test.loadImgs(img_ids[i])
                        I = io.imread('%s/images/%s/%s' % ('./data', 'test', img[0]['file_name']))
                        plt.axis('off')
                        plt.imshow(I)
                        plt.savefig('./captions/{}/bad_images/{}.png'.format(self.__name, img_ids[i]))
                        joined_reference_captions = []
                        for caption in reference_captions:
                            joined_reference_captions.append(' '.join(caption))

                        low_temperature_caption = self.__model.sample(images[i, :, :, :].unsqueeze(0), self.__vocab
                                                                      , self.__generation_config['max_length']
                                                                      , deterministic=False
                                                                      , temp=0.001)[0]
                        high_temperature_caption = self.__model.sample(images[i, :, :, :].unsqueeze(0), self.__vocab
                                                                       , self.__generation_config['max_length']
                                                                       , deterministic=False
                                                                       , temp=5)[0]
                        deterministic_caption = self.__model.sample(images[i, :, :, :].unsqueeze(0), self.__vocab
                                                                    , self.__generation_config['max_length']
                                                                    , deterministic=True
                                                                    , temp=0.4)[0]
                        with open('./captions/{}/bad_images/{}_captions.txt'.format(self.__name, img_ids[i]),
                                  'w') as f:
                            f.write('Actual captions:\n')
                            for caption in joined_reference_captions:
                                f.write(caption)
                                f.write('\n')
                            f.write('Predicted caption:\n')
                            f.write(' '.join(pred_captions))
                            f.write('\n')
                            f.write('Predicted caption with 0.001 temperature:\n')
                            f.write(' '.join(low_temperature_caption))
                            f.write('\n')
                            f.write('Predicted caption with 5 temperature:\n')
                            f.write(' '.join(high_temperature_caption))
                            f.write('\n')
                            f.write('Predicted caption with deterministic:\n')
                            f.write(' '.join(deterministic_caption))
                            f.write('\n')






    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        # plt.show()
