# Music Generation Using LSTM

This project focuses on generating music using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. It trains on MIDI data and produces new compositions by learning patterns in the dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

## Introduction
Music generation using deep learning is an exciting application of AI. This project implements an LSTM-based neural network to generate music sequences in MIDI format. The model is trained on a dataset of MIDI files, learning patterns in musical notes and rhythms to produce coherent melodies.

## Dataset
The dataset consists of MIDI files that have been converted into a numerical format for training. The `midi_data.csv` file contains the extracted musical notes and durations.

## Model Architecture
The model consists of:
- An embedding layer to process input sequences
- Multiple LSTM layers for learning long-term dependencies in the data
- A dense output layer with softmax activation for predicting the next note

The trained model is stored as `music.h5` and can generate new MIDI compositions.

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy tensorflow music21 matplotlib
```

## Usage
### Training the Model
Run the `music_generation.ipynb` notebook to train the model on the provided MIDI dataset. Training checkpoints are stored in the `training_checkpoints/` directory.

### Generating Music
Once trained, you can generate music using:
```python
python generate_music.py
```
This will produce an output MIDI file (`output.midi`).

## Results
After training, the model can generate melodies that resemble the patterns in the training data. The generated MIDI files can be played using any MIDI player or DAW (Digital Audio Workstation).

## Directory Structure
```
.
├── training_checkpoints/    # Stores model checkpoints
├── .gitignore               # Files to ignore in Git
├── LICENSE                  # Project license
├── README.md                # Project documentation
├── midi_data.csv            # Processed MIDI dataset
├── music.h5                 # Trained LSTM model
├── music_generation.ipynb   # Jupyter Notebook for training
├── output.midi              # Generated music output
```

## Future Improvements
- Implement attention mechanisms to improve long-term dependencies.
- Train on a larger dataset for better generalization.
- Fine-tune the hyperparameters for better melody coherence.
- Experiment with transformer-based models for music generation.

## Acknowledgments
This project utilizes the `music21` library for MIDI processing and TensorFlow for training LSTM networks.
