import os
import random
import numpy as np
import librosa
import torch
from pydub import AudioSegment

root_dir = str(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def create_dir(dir: str) -> None:
    """
    Creates a directory if it does not already exist.

    Args:
        dir (str): Path to the directory to be created.

    Returns:
        None
    """
    if os.path.isdir(dir):
        print(dir, 'already exists. Continuing ...' )
    else:
         print('Creating new dir: ', dir)
         os.makedirs(dir)

 # convert .flac files to .wav files in the voice data folder
def flac2wav() -> None:
    """
    Converts all `.flac` audio files in the voice data folder to `.wav` format.

    - Reads `.flac` files from the `voice_detect/data/voice/flac/` directory.
    - Converts them to `.wav` format and saves them in the `voice_detect/data/voice/wav/` directory.

    Args:
        None

    Returns:
        None
    """
    flac_path = str(root_dir + '/voice_detect/data/voice/flac/')
    wav_path = str(root_dir + '/voice_detect/data/voice/wav/')
    flac_files = [f for f in os.listdir(flac_path) if os.path.isfile(os.path.join(flac_path, f)) and f.endswith('.flac')]

    for file in flac_files:
        print('Converting ' + str(file))
        temp = AudioSegment.from_file(str(flac_path + file))
        temp.export(str(wav_path + os.path.splitext(file)[0]) + '.wav', format='wav')
    print('Done converting \n')


# create the training and testing split lists for both classes
def create_splits(voice_path: str, not_voice_path: str) -> list:
    """
    Creates training and testing splits for voice and non-voice datasets.

    - Splits the datasets into an 80/20 ratio for training and testing.
    - Combines voice and non-voice splits into two complete lists.

    Args:
        voice_path (str): Path to the directory containing voice `.wav` files.
        not_voice_path (str): Path to the directory containing non-voice `.wav` files.

    Returns:
        list: A tuple containing two lists:
            - full_train_list: List of training file paths.
            - full_test_list: List of testing file paths.
    """
    voice_wavs = str(root_dir + '/voice_detect/data/voice/wav/')
    not_voice_wavs = str(root_dir + '/voice_detect/data/not_voice/wav/')

    # get total number of files in the both dirs and split the training and testing dataset    
    # by a 80/20 ratio
    voice_list = [voice_wavs + name for name in os.listdir(voice_wavs)]
    voice_total  = len(voice_list)
    voice_train_split = round(voice_total * 0.8)
    voice_test_split = voice_total - voice_train_split

    assert voice_train_split + voice_test_split == voice_total

    voice_train_list = random.sample(voice_list, voice_train_split)
    voice_test_list = random.sample(voice_list, voice_test_split)

    not_voice_list = [not_voice_wavs + name for name in os.listdir(not_voice_wavs)]
    not_voice_total  = len(not_voice_list)
    not_voice_train_split = round(not_voice_total * 0.8)
    not_voice_test_split = not_voice_total - not_voice_train_split

    assert not_voice_train_split + not_voice_test_split == not_voice_total

    not_voice_train_list = random.sample(not_voice_list, not_voice_train_split)
    not_voice_test_list = random.sample(not_voice_list, not_voice_test_split)

    # concat into two complete lists
    full_train_list = voice_train_list + not_voice_train_list
    full_test_list = voice_test_list + not_voice_test_list
    
    return full_train_list, full_test_list


# calculate accuracy of a prediction
def get_accuracy(prediction: str, label: str) -> float:
    """
    Calculates the accuracy of a model's predictions.

    Args:
        prediction (str): Predicted labels as a tensor.
        label (str): Ground truth labels as a tensor.

    Returns:
        float: Accuracy as a percentage of correct predictions.
    """
    matches  = [torch.argmax(i) == torch.argmax(j) for i, j in zip(prediction, label)]
    accuracy = matches.count(True) / len(matches)
    return accuracy
