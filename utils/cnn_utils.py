import os
import re
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from utils.gen_utils import create_dir, create_splits

root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def get_melss(wav_file: str, new_name: str) -> None:
    """
    Generates and saves a mel-spectrogram image from a given WAV file.

    Args:
        wav_file (str): Path to the input WAV file.
        new_name (str): Path to save the generated mel-spectrogram image (JPG format).

    Returns:
        None
    """
    # get sample rate
    x, sr = librosa.load(wav_file, sr=None, res_type='kaiser_fast')

    # get headless figure
    fig = plt.figure(figsize=[1, 1])
    
    # remove the axes
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    # get melss
    melss = librosa.feature.melspectrogram(y=x, sr=sr)
    librosa.display.specshow(librosa.power_to_db(melss, ref=np.max), y_axis='linear')
    
    # save plot as jpg
    plt.savefig(new_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


def prepare_dataset() -> None:
    """
    Prepares the dataset for CNN training and testing by generating mel-spectrogram images
    for audio files in the dataset.

    - Splits the dataset into training and testing sets (80/20 split).
    - Creates necessary directories for storing spectrogram images.
    - Converts audio files into mel-spectrogram images and saves them in the appropriate directories.

    Args:
        None

    Returns:
        None
    """
    # get training and testing splits
    voice = os.path.join('/opt/voice_detect/data/voice/')
    not_voice = os.path.join('/opt/voice_detect/data/not_voice/')
    train, test = create_splits(voice, not_voice)

    voice_train = os.path.join('/opt/voice_detect/data/plots/train/voice/')
    not_voice_train = os.path.join('/opt/voice_detect/data/plots/train/not_voice/')
    voice_test = os.path.join('/opt/voice_detect/data/plots/test/voice/')
    not_voice_test = os.path.join('/opt/voice_detect/data/plots/test/not_voice/')
    
    create_dir(voice_train)
    create_dir(not_voice_train)
    create_dir(voice_test)
    create_dir(not_voice_test)

    # iterate through the training split
    for file in train:
        try:
            print('Making train plot for: ' + file)
            if 'not_voice' in file:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)
                
                # construct new jpg file name with the extenstion
                jpg_file_name = str(wav_name[0]) + '.jpg'
                jpg_file_name = str(not_voice_train + jpg_file_name)
                get_melss(file, jpg_file_name)
            else:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)
                
                # construct new jpg file name with the extenstion
                jpg_file_name = str(wav_name[0]) + '.jpg'
                jpg_file_name = str(voice_train + jpg_file_name)
                get_melss(file, jpg_file_name)

        except Exception:
            print('ERROR at ' + file + ' CONTINUING ...')
            pass
            
    # iterate through the testing split
    for file in test:
        try:
            print('Making test plot for: ' + file)
            if 'not_voice' in file:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)
                
                # construct new jpg file name with the extenstion
                jpg_file_name = str(wav_name[0]) + '.jpg'
                jpg_file_name = str(not_voice_test + jpg_file_name)
                get_melss(file, jpg_file_name)
            else:
                wav_name = os.path.basename(file)
                wav_name = os.path.splitext(wav_name)
                
                # construct new jpg file name with the extenstion
                jpg_file_name = str(wav_name[0]) + '.jpg'
                jpg_file_name = str(voice_test + jpg_file_name)
                get_melss(file, jpg_file_name)

        except Exception:
            print('ERROR at ' + file + ' CONTINUING ...')
            pass
grep -rnw '/path/to/somewhere/' -e 'pattern'