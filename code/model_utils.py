import numpy as np
import torch

def get_next_word_idx(prediction_lst, config):
    deterministic = config['generation']['deterministic']
    temperature = config['generation']['temperature']
    if deterministic is True:
        return prediction_lst.argmax(1).item()

    else:
        prediction_lst = np.exp(prediction_lst/temperature) / np.exp(prediction_lst/temperature).sum(axis=1)
        return np.random.choice(range(len(prediction_lst)), p=prediction_lst)