import shutil
import numpy as np
import matplotlib.pyplot as plt
import os, re
import matplotlib
matplotlib.use('Agg')
import pylab
import librosa
import librosa.display


dir='../../IEMOCAP_audio/'
all_emotions=os.listdir(dir)
for each_emotion in all_emotions:
    each_emotion_dir=os.path.join(dir, each_emotion)
    all_audio=os.listdir(each_emotion_dir)
    for each_audio in all_audio:
        each_audio_dir = os.path.join(each_emotion_dir, each_audio)
        sig, fs = librosa.load(each_audio_dir)
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        S = librosa.feature.melspectrogram(y=sig, sr=16000, n_fft=512, hop_length=160, fmax=16000/5)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        if os.path.exists('./data/ravdess/train' + each_emotion) is False:
            os.makedirs('./data/ravdess/train' + each_emotion)
        pylab.savefig(
            './data/ravdess/train' + each_emotion+ '/' + each_audio.split('.wav')[0] + '.jpg',
            bbox_inches=None, pad_inches=0)
        pylab.close()