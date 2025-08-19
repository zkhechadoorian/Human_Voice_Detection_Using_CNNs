# 📊 Project Title: Voice Detection

In an increasingly audio-centric digital world, distinguishing human voice from other environmental sounds is critical. Whether in smart devices, surveillance systems, customer service bots, or accessibility tools, the ability to accurately detect human voice in real-time audio feeds provides a competitive advantage.


This project delivers a **robust AI-driven Voice Detection System** that classifies audio clips into either **voice** or **non-voice** categories. It leverages deep learning technique **Convolutional Neural Networks (CNN)**—to build a high-accuracy binary classification model.
Based on the directory listings you've provided, here's a clean and structured overview of your **project structure** for **Human Voice Detection**:

---

## **Project Structure: `human-voice-detection/`**

```
human-voice-detection/
├── README.md                 # Project overview and documentation
├── requirements.txt          # Python dependencies
├── flac2wav.sh               # Script to convert FLAC audio to WAV format
├── prep_cnn_data.sh          # Script to prepare data for CNN (generate spectrograms)
├── saved_models/             # Directory to store trained model files
├── cnn/                      # Convolutional Neural Network code
│   ├── train.py              # CNN training script
│   ├── eval.py               # Evaluation script for CNN
│   ├── model.py              # CNN architecture definition
├── utils/                    # Utility functions
│   ├── __init__.py           # Marks utils as a Python package
│   ├── cnn_utils.py          # CNN-specific utilities (e.g., image processing)
│   └── gen_utils.py          # General utility functions (e.g., file handling, feature extraction)
├── voice_detect/             # Likely contains FFNN model code (not fully listed)
```

---

## 🔧 Setup and Installation Instructions


### 1. Create a virtual environment and activate it

```bash
python3.8 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
---

## 🛠️ Step-by-Step Guide

### Step-01

DO NOT RUN IT WITHOUT READING THE FOLLOWING NOTES
---

```bash
./flac2wav.sh 
```
```bash
./prep_cnn_data.sh 
```
---

This utility script supports the voice detection project by handling essential data preparation task. It includes functions to convert `.flac` audio files to the more commonly used `.wav` format, create train-test splits from voice and non-voice datasets using an 80/20 ratio, and compute model accuracy by comparing predictions with ground truth labels. Additionally, it ensures necessary directories are created before training or saving outputs. These utilities streamline the preprocessing pipeline, making the training process for both the CNN model efficient and reproducible. However, due to the large size of the original dataset, all preprocessing has already been completed. As a result, there is no need to re-run these scripts. If executed again, they may produce errors—not because the code is broken, but simply because the data has already been processed, cleaned, and organized into its final structure.

### Step-02

```bash
python -m cnn.train
```

---

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify audio spectrogram images into two categories: **voice** or **not voice**. The spectrograms, preprocessed from audio data and stored in a folder-based dataset, are resized to 32×32 and transformed into tensors using `torchvision.transforms`. The CNN model architecture consists of three convolutional layers with batch normalization, max pooling, and dropout, followed by four fully connected layers to perform the final classification. The training loop runs for 250 epochs using the Adam optimizer and cross-entropy loss, with progress logged and the trained model saved after each epoch. Accuracy is calculated using a custom utility function, and all models are saved in a structured directory by date. This system is part of a larger voice detection pipeline that combines this CNN with a feedforward neural network for robust ensemble inference on real-world audio data.

---

### Step-03

> **Note:** You can replace `train-08-02-2025/CNN-08-02-2025.pt` with the path to any other trained CNN model file.

```bash
python -m cnn.eval CNN/train-08-02-2025/CNN-08-02-2025.pt
```
---

This script evaluates a trained Convolutional Neural Network (CNN) on the test dataset of spectrogram images to determine how well the model distinguishes between **voice** and **not voice** audio recordings. It loads the specified `.pt` model file, applies standardized transforms to the test data, and uses the model to generate predictions. The script then calculates and prints detailed performance metrics, including overall accuracy, a confusion matrix, per-class accuracy, and a full classification report. This evaluation step ensures that the CNN model generalizes well to unseen data and serves as a critical checkpoint in the voice detection pipeline. **Note:** You can replace the model path with any other trained CNN checkpoint.

---

### 📊 Results

After running the evaluation script, you will see several important metrics printed in the terminal that help you understand how well the CNN model performs on unseen test data.

* **Accuracy Score**: This is the overall classification accuracy, showing the percentage of correctly predicted samples. In this case, `0.9966` means the model correctly predicted \~99.66% of the test samples.

* **Confusion Matrix**: This table breaks down the number of correct and incorrect predictions for each class.

  * The format is:

    ```
    [[True Negatives, False Positives],
     [False Negatives, True Positives]]
    ```
  * For example, `[[1735, 11], [1, 1833]]` means:

    * 1735 non-voice clips were correctly predicted as non-voice.
    * 11 non-voice clips were incorrectly predicted as voice.
    * 1 voice clip was incorrectly predicted as non-voice.
    * 1833 voice clips were correctly predicted as voice.

* **Per-Class Accuracy**: This shows the classification accuracy for each individual class.

  * Class `0` (non-voice): 99.37% accuracy
  * Class `1` (voice): 99.95% accuracy

* **Classification Report**: Provides precision, recall, and F1-score for each class. These metrics help assess how balanced and reliable the model is:

  * **Precision**: Of the predicted instances of a class, how many were actually correct.
  * **Recall**: Of all actual instances of a class, how many were correctly predicted.
  * **F1-score**: The harmonic mean of precision and recall.
  * **Support**: The number of actual occurrences for each class in the test set.

These results are printed directly in the terminal. There are no output files generated by this script, but the model used for evaluation (`CNN/train-08-02-2025/CNN-08-02-2025.pt`) is stored in the `saved_models/` directory, along with a `.log` file from training that contains similar metrics over training epochs.
