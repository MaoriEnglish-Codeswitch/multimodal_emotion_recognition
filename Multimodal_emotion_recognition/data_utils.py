import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def get_text_video_audio_data(data_path, part='train'):
    if part == 'train':
        x_txt = np.load(data_path+'/'+'train_text.npy')
        x_vid = np.load(data_path+'/'+'train_video.npy')
        x_aud = np.load(data_path+'/'+'train_audio.npy')
        labels = np.load(data_path+'/'+'train_label.npy')

    elif part == 'dev':
        x_txt = np.load(data_path + '/' + 'dev_text.npy')
        x_vid = np.load(data_path+'/'+'dev_video.npy')
        x_aud = np.load(data_path + '/' + 'dev_audio.npy')
        labels = np.load(data_path+'/'+'dev_label.npy')

    elif part == 'test':
        x_txt = np.load(data_path + '/' + 'test_text.npy')
        x_vid = np.load(data_path+'/'+'test_video.npy')
        x_aud = np.load(data_path + '/' + 'test_audio.npy')
        labels = np.load(data_path+'/'+'test_label.npy')

    return x_txt, x_vid, x_aud, labels

def save_model(model, name):
    torch.save(model, name)

def load_model(name):
    model = torch.load(name)
    return model

# taken from https://github.com/david-yoon/attentive-modality-hopping-for-SER
'''
list_y_ture : reference (label)
list_y_pred : predicted value
note        : do not consider "label imbalance"
'''


def unweighted_accuracy(list_y_true, list_y_pred):
    assert (len(list_y_true) == len(list_y_pred))

    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)

    return accuracy_score(y_true=y_true, y_pred=y_pred)


'''
list_y_ture : reference (label)
list_y_pred : predicted value
note        : compute accuracy for each class; then, average the computed accurcies
              consider "label imbalance"
'''


def weighted_accuracy(list_y_true, list_y_pred):
    assert (len(list_y_true) == len(list_y_pred))

    y_true = np.array(list_y_true)
    y_pred = np.array(list_y_pred)

    w = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w[y_true == idx] = float(1 / i)

    return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=w)

def weighted_precision(list_y_true, list_y_pred):
    wa = precision_score(y_true=list_y_true, y_pred=list_y_pred, average='weighted')
    return wa

def unweighted_precision(list_y_true, list_y_pred):
    uwa = precision_score(y_true=list_y_true, y_pred=list_y_pred, average='macro')
    return uwa
