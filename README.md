# Speech Confidence Classification

This repository contains a speech confidence classification pipeline. The pipeline includes feature extraction from audio, clustering based on confidence, and building a BiLSTM, CNN and RNN neural network model for classification. The aim is to classify audio segments based on their confidence levels, aiding in the analysis and processing of speech data.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run](#to-run)

## Project Structure

project
    |-> BiLSTM
    |   |-> main.py
    |-> CNN
    |   |-> main.py
    |-> RNN
    |   |-> main.py
    |-> speech_data_wav
    |   |-> train
    |   |   |-> labels
    |   |   |-> wav
    |   |-> test
    |   |-> output_labels

## Installation

git clone https://github.com/username/speech_recognition_project.git
cd speech_recognition_project

### To install required packages
pip install -r requirements.txt

## To run
python bilstm/main.py
python cnn/main.py
python rnn/main.py