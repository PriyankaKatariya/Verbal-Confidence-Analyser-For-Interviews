import os
import pandas as pd
import numpy as np
import librosa
import pyaudio
import wave
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

# Step 1: Read speech files
def read_speech_files(file_paths):
    speech_vectors = []
    for file_path in file_paths:
        speech, _ = librosa.load(file_path, sr=None)
        speech_vectors.append(speech)
    return speech_vectors

# Step 2: Parse annotation files
def parse_annotation_file(annotation_file):
    segments = []
    with open(annotation_file, 'r') as file:
        for line in file:
            start_time, end_time, confidence = line.strip().split()
            segments.append((float(start_time), float(end_time), confidence))
    return segments

# Step 3: Create feature vectors
def create_feature_vectors(speech_vectors, annotation_files, sr, feature_type='mfcc'):
    feature_vectors = []
    for i, speech_vector in enumerate(speech_vectors):
        segments = parse_annotation_file(annotation_files[i])
        for segment in segments:
            start_time, end_time, confidence = segment
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            segment_features = speech_vector[start_frame:end_frame]
            if len(segment_features) > 0:  # Check if segment has non-zero length
                if feature_type == 'mfcc':
                    features = librosa.feature.mfcc(segment_features, sr=sr, n_mfcc=13)
                elif feature_type == 'chroma':
                    stft = np.abs(librosa.stft(segment_features))
                    features = librosa.feature.chroma_stft(S=stft, sr=sr)
                feature_vector = np.mean(features, axis=1)
                feature_vectors.append((feature_vector, confidence))
    return feature_vectors

# Step 4: Apply K-means clustering
def apply_clustering(feature_vectors):
    X = np.array([segment_features for segment_features, _ in feature_vectors])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    return kmeans.labels_, X

# Step 5: Visualize clusters using PCA
def visualize_clusters(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Clustering based on Confidence Level - MFCC')
    plt.colorbar(label='Cluster')
    plt.show()
    

# Step 6: Get file paths from directories
def get_file_paths(directory, extension):
    file_paths = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_paths.append(os.path.join(directory, file))
    return file_paths

# Example usage
confidence_levels = ['h', 'm', 'l']
speech_directory = 'project//speech_data_wav//train//wav'
annotation_directory = 'project//speech_data_wav//train//labels'
sr = 44100  # Sample rate

# Get file paths
speech_files = get_file_paths(speech_directory, '.wav')
annotation_files = get_file_paths(annotation_directory, '.txt')

# Ensure the corresponding annotation file exists for each speech file
speech_files.sort()
annotation_files.sort()

# Read speech files
speech_vectors = read_speech_files(speech_files)

# Create feature vectors with MFCC
feature_vectors_mfcc = create_feature_vectors(speech_vectors, annotation_files, sr, feature_type='mfcc')

# Create feature vectors with Chroma
feature_vectors_chroma = create_feature_vectors(speech_vectors, annotation_files, sr, feature_type='chroma')

# Apply clustering
labels_mfcc, X_mfcc = apply_clustering(feature_vectors_mfcc)
labels_chroma, X_chroma = apply_clustering(feature_vectors_chroma)

# Visualize clusters
visualize_clusters(X_mfcc, labels_mfcc)
visualize_clusters(X_chroma, labels_chroma)

def txt2df(file):
    """
    Read a text file and return a DataFrame with 'start', 'end', and 'confidence' columns.
    Args:
        file (str): Path to the text file.
    Returns:
        pd.DataFrame: DataFrame containing the parsed data.
    """
    try:
        df = pd.read_csv(file, sep='\t', header=None, names=['start', 'end', 'confidence'])
        confidence_mapping = {'l': 0, 'm': 1, 'h': 2}
        df['confidence'] = df['confidence'].map(confidence_mapping)
        df['start'] = df['start'].astype(float)
        df['end'] = df['end'].astype(float)
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")
    return df

def load_audio(file):
    """
    Load an audio file.
    Args:
        file (str): Path to the audio file.
    Returns:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.
    """
    y, sr = librosa.load(file, sr=None)
    return y, sr

def extract_features(segments, sr, feature_type='mfcc', n_mfcc=13):
    """
    Extract features from audio segments.
    Args:
        segments (list): List of tuples (segment, confidence).
        sr (int): Sampling rate.
        feature_type (str): Feature extraction type ('mfcc' or 'chroma').
        n_mfcc (int): Number of MFCCs to return (if feature_type is 'mfcc').
    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
    """
    X = []
    y = []
    for segment, confidence, start, end in segments:
        if feature_type == 'mfcc':
            features = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        elif feature_type == 'chroma':
            features = librosa.feature.chroma_stft(y=segment, sr=sr)
        features = np.mean(features.T, axis=0)
        X.append(features)
        y.append(confidence)
    return np.array(X), np.array(y)

# without any modification
# def train_model(X_train, y_train):
#     """
#     Train a CNN model and plot loss and accuracy.

#     Args:
#         X_train (np.ndarray): Training features.
#         y_train (np.ndarray): Training labels.

#     Returns:
#         model: Trained CNN model.
#         history: Training history.
#     """
#     model = Sequential()
#     model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Conv2D(64, (3, 1), activation='relu'))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2)
#     return model, history

# with L2 regularisation
# def train_model_l2(X_train, y_train, l2_penalty=0.001):
#     """
#     Train a CNN model with L2 regularization and plot loss and accuracy.

#     Args:
#         X_train (np.ndarray): Training features.
#         y_train (np.ndarray): Training labels.
#         l2_penalty (float): L2 regularization penalty.

#     Returns:
#         model: Trained CNN model.
#         history: Training history.
#     """
#     model = Sequential()
#     model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1),
#                      kernel_regularizer=l2(l2_penalty)))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Conv2D(64, (3, 1), activation='relu', kernel_regularizer=l2(l2_penalty)))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_penalty)))
#     model.add(Dense(3, activation='softmax', kernel_regularizer=l2(l2_penalty)))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2)
#     return model, history

# with Dropout 0.2
# def train_model_dropout(X_train, y_train, dropout_rate=0.2):
#     """
#     Train a CNN model with dropout regularization and plot loss and accuracy.

#     Args:
#         X_train (np.ndarray): Training features.
#         y_train (np.ndarray): Training labels.
#         dropout_rate (float): Dropout rate.

#     Returns:
#         model: Trained CNN model.
#         history: Training history.
#     """
#     model = Sequential()
#     model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Dropout(dropout_rate))  # Adding dropout after the first convolutional layer
#     model.add(Conv2D(64, (3, 1), activation='relu'))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(dropout_rate))  # Adding dropout after the dense layer
#     model.add(Dense(3, activation='softmax'))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2)
#     return model, history

# with dropout and L2
# def train_model(X_train, y_train, l2_penalty=0.001, dropout_rate=0.2):
#     """
#     Train a CNN model with both L2 regularization and dropout regularization and plot loss and accuracy.

#     Args:
#         X_train (np.ndarray): Training features.
#         y_train (np.ndarray): Training labels.
#         l2_penalty (float): L2 regularization penalty.
#         dropout_rate (float): Dropout rate.

#     Returns:
#         model: Trained CNN model.
#         history: Training history.
#     """
#     model = Sequential()
#     model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1),
#                      kernel_regularizer=l2(l2_penalty)))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Dropout(dropout_rate))  # Adding dropout after the first convolutional layer
#     model.add(Conv2D(64, (3, 1), activation='relu', kernel_regularizer=l2(l2_penalty)))
#     model.add(MaxPooling2D((2, 1)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_penalty)))
#     model.add(Dropout(dropout_rate))  # Adding dropout after the dense layer
#     model.add(Dense(3, activation='softmax', kernel_regularizer=l2(l2_penalty)))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2)
#     return model, history

#CNN
def train_model(X_train, y_train, l2_penalty=0.001, dropout_rate=0.2, num_filters=(32, 64), learning_rate=0.0001, epochs=80):
    """
    Train a CNN model with further adjustments to hyperparameters and plot loss and accuracy.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        l2_penalty (float): L2 regularization penalty.
        dropout_rate (float): Dropout rate.
        num_filters (tuple): Number of filters for each convolutional layer.
        learning_rate (float): Learning rate for the Adam optimizer.
        epochs (int): Number of training epochs.

    Returns:
        model: Trained CNN model.
        history: Training history.
    """
    model = Sequential()
    model.add(Conv2D(num_filters[0], (3, 1), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1),
                     kernel_regularizer=l2(l2_penalty)))
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(dropout_rate))  # Adding dropout after the first convolutional layer
    model.add(Conv2D(num_filters[1], (3, 1), activation='relu', kernel_regularizer=l2(l2_penalty)))
    model.add(MaxPooling2D((2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_penalty)))
    model.add(Dropout(dropout_rate))  # Adding dropout after the dense layer
    model.add(Dense(3, activation='softmax', kernel_regularizer=l2(l2_penalty)))
    
    optimizer = Adam(learning_rate=learning_rate)  # Adjust the learning rate
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
    return model, history

def predict_confidence(model, scaler, file, output_dir, feature_type='mfcc'):
    """
    Predicts the confidence level of segments in an audio file and writes the results to a text file.

    Parameters:
    model (keras.Model): Trained machine learning model for prediction.
    scaler (sklearn.preprocessing.StandardScaler): Scaler used to normalize the feature vectors.
    file (str): Path to the input audio file.
    output_dir (str): Directory where the output text file will be saved.
    feature_type (str): Feature extraction type ('mfcc' or 'chroma').
    """
    y, sr = load_audio(file)
    segment_duration = 15.0  # Duration of each segment in seconds
    n_samples_per_segment = int(segment_duration * sr)
    
    segments = [(y[i:i + n_samples_per_segment], None, None, None) for i in range(0, len(y), n_samples_per_segment)]
    X_test, _ = extract_features(segments, sr, feature_type=feature_type)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_pred = model.predict(X_test)
    
    # Rounding predictions to the nearest integer
    y_pred = np.argmax(y_pred, axis=1)
    
    # Generating output label file
    base_filename = os.path.basename(file)
    label_filename = os.path.join(output_dir, base_filename.replace('.wav', '.txt'))
    
    with open(label_filename, 'w') as f:
        for i, confidence in enumerate(y_pred):
            start_time = i * segment_duration
            end_time = start_time + segment_duration
            confidence_label = ['l', 'm', 'h'][confidence]
            f.write(f"{start_time:.2f}\t{end_time:.2f}\t{confidence_label}\n")

# Main Function to Execute All Steps
def main(training_dir, feature_type='mfcc'):
    """
    Main function to execute all steps.
    Args:
        training_dir (str): Directory containing training audio and label files.
        feature_type (str): Feature extraction type ('mfcc' or 'chroma').
    """
    X = []
    y = []
    times = []

    # Processing Training Data
    audio_dir = os.path.join(training_dir, 'wav')
    label_dir = os.path.join(training_dir, 'labels')

    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            audio_file = os.path.join(audio_dir, filename)
            label_file = os.path.join(label_dir, filename.replace('.wav', '.txt'))
            y_audio, sr = load_audio(audio_file)
            df_labels = txt2df(label_file)
            segments = [(y_audio[int(start * sr):int(end * sr)], confidence, start, end) for start, end, confidence in zip(df_labels['start'], df_labels['end'], df_labels['confidence'])]
            X_features, y_labels = extract_features(segments, sr, feature_type=feature_type)
            if X_features.size > 0 and y_labels.size > 0:  # Check if the arrays are non-empty
                X.extend(X_features)
                y.extend(y_labels)
                times.extend([(start, end) for start, end, _, _ in segments])

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    if X.size == 0 or y.size == 0:
        raise ValueError("No features or labels were extracted. Please check the data and feature extraction process.")

    # Train-Test Split for Model Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training feature matrix shape: {X_train.shape}")
    print(f"Testing feature matrix shape: {X_test.shape}")

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape features for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  # shape: (num_samples, feature_dim, 1, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)  # shape: (num_samples, feature_dim, 1, 1)

    print(f"Reshaped training feature matrix shape: {X_train.shape}")
    print(f"Reshaped testing feature matrix shape: {X_test.shape}")

    # Training the Model
    model, history = train_model(X_train, y_train)

    # Evaluating the Model
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    # Plotting loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()
    return model



def record_audio(output_filename, record_seconds=20, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
    """
    Records audio from the default input device and saves it to a WAV file.

    Parameters:
    output_filename (str): The name of the output WAV file.
    record_seconds (int): The duration of the recording in seconds. Default is 20 seconds.
    format (pyaudio format): The format of the audio data. Default is pyaudio.paInt16.
    channels (int): The number of audio channels. Default is 1.
    rate (int): The sampling rate in Hz. Default is 44100.
    chunk (int): The number of frames per buffer. Default is 1024.
    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording for the next", record_seconds, "seconds...")
    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print("Audio saved as:", output_filename)

training_dir = 'project//speech_data_wav//train'
testing_dir = 'project//speech_data_wav//test'
output_dir = 'project//speech_data_wav//output_labels'

print("Training and testing with MFCC features:")
model_mfcc = main(training_dir, feature_type='mfcc')

print("Training and testing with Chroma features:")
model_chroma = main(training_dir, feature_type='chroma')