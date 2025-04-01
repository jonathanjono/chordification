import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import time
from scipy.ndimage import maximum_filter1d
import sounddevice as sd
import queue
import pyttsx3
import speech_recognition as sr
import flet as ft
import threading

# text2speech
engine = pyttsx3.init()
engine.setProperty('volume', 0.5)

sample_rate = 16000
dt = 1.0
n_mels = 64
hop_length = 256
n_fft = 1024
frames = (sample_rate // hop_length) + 1
N_CLASSES = 8

#------------------
keys_dictionary = {
    "C": 0,
    "C#": -1, "Db": -1,
    "D": -2,
    "D#": -3, "Eb": -3,
    "E": -4,
    "F": -5,
    "F#": 6, "Gb": 6,
    "G": 5,
    "G#": 4, "Ab": 4,
    "A": 3,
    "A#": 2, "Bb": 2,
    "B": 1
}

#------------------
note_to_index = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10, "B flat": 10,
    "B": 11
}

#------------------
index_to_note = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}

#------------------
audio_queue = queue.Queue()


def get_mode():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say the mode (continuous or learn): ")
        engine.say("Enter mode, continuous or learn")
        engine.runAndWait()
        try:
            mode = recognizer.listen(source, timeout=5)
            mode = recognizer.recognize_google(mode).lower()
            print(f"Mode input: {mode}")
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            mode = ""

    while mode not in ["continuous", "learn"]:
        print("Please enter mode (continuous or learn): ")
        engine.say("Please enter mode, continuous or learn")
        engine.runAndWait()
        with sr.Microphone() as source:
            try:
                mode = recognizer.listen(source, timeout=5)
                mode = recognizer.recognize_google(mode).lower()
                print(f"Mode input: {mode}")
            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                mode = ""

    return mode


def callback(indata: np.ndarray, frames, time, status) -> None: # calls everytime whenvever there is new data in stream
    if status:
        print(status)
    audio_queue.put(indata.copy())


def envelope(y):
    window_size = int(sample_rate / 20)
    y_abs = np.abs(y)
    y_max = maximum_filter1d(y_abs, size=window_size, mode='reflect')
    mask = y_max > 0.01
    return mask


def get_melspectrogram(waveform):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel_spectrogram)


def get_key():
    recognizer = sr.Recognizer()
    key_mappings = {
        "see sharp": "C#",
        "see": "C",
        "d flat": "Db",
        "d sharp": "D#",
        "e flat": "Eb",
        "f sharp": "F#",
        "g flat": "Gb",
        "g sharp": "G#",
        "a flat": "Ab",
        "a sharp": "A#",
        "b flat": "Bb"
    }

    mic = sr.Microphone()

    while True:
        print("Say key name (major keys only such as C, C#, D, etc. with no flats): ")
        engine.say("Say the key name")
        engine.runAndWait()

        try:
            mic_stream = mic.open()
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            audio = recognizer.listen(mic, timeout=10)
            mic_stream.close()

            key = recognizer.recognize_google(audio).lower()
            key = key.split()[0].capitalize()
            key = key_mappings.get(key.lower(), key)

            print(f"Key input: {key}")

            if key in keys_dictionary:
                return key
            else:
                print("invalid key, please try again.")
                engine.say("invalid key, please try again.")
                engine.runAndWait()

        except sr.UnknownValueError:
            print("Could not understand audio, please try again.")
            engine.say("Could not understand audio, please try again.")
            engine.runAndWait()
        except sr.RequestError:
            print("Failed. Please check your internet connection.")
            engine.say("Failed. Please check your internet connection.")
            engine.runAndWait()
        except sr.WaitTimeoutError:
            print("No speech detected, please try again.")
            engine.say("No speech detected, please try again.")
            engine.runAndWait()


def transpose_audio(audio_array, key):
    semitones = keys_dictionary[key]
    if semitones == 0:
        return audio_array, semitones
    y_transpose = librosa.effects.pitch_shift(audio_array, sr=sample_rate, n_steps=semitones)
    return y_transpose, semitones


def predict_class(learn, transposed_audio, interpreter, le, input_details, output_index):
    mask = envelope(transposed_audio)
    y = transposed_audio[mask]
    n = y.shape[0]
    m = n // sample_rate

    predictions = []
    if not learn:
        for k in range(m):
            sample = y
            if len(sample) < sample_rate:
                sample = np.pad(sample, (0, sample_rate - len(sample)))
            mel_spec = get_melspectrogram(sample)
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]

            interpreter.set_tensor(input_details[0]['index'], mel_spec)
            interpreter.invoke()

            pred = interpreter.get_tensor(output_index)
            pred_class = np.argmax(pred, axis=-1).item()
            predictions.append(pred_class)
    if learn:
        for k in range(m):
            sample = y[k * sample_rate:(k + 1) * sample_rate]
            if len(sample) < sample_rate:
                # pad
                sample = np.pad(sample, (0, sample_rate - len(sample)))
            mel_spec = get_melspectrogram(sample)
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]

            interpreter.set_tensor(input_details[0]['index'], mel_spec)
            interpreter.invoke()

            pred = interpreter.get_tensor(output_index)
            pred_class = np.argmax(pred, axis=-1).item()
            predictions.append(pred_class)

    if len(predictions) <= 0:
        return None
    else:
        final_prediction = np.bincount(predictions).argmax()
        return le.inverse_transform([final_prediction])[0]
stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=1024, latency='high')

def transpose_back_chord(chord_name, semitone_shift):
    chord_text = chord_name.split()
    transposed_chord = chord_text[0]
    quality = chord_text[1]
    predicted_index = note_to_index[transposed_chord]
    true_index = (predicted_index - semitone_shift) % 12  # reverse the semitone shift and mod makes sure its in the cycle
    new_root = index_to_note[true_index]
    if quality:
        return f"{new_root} {quality}"
    else:
        return new_root


def run_flet(chord_queue):
    def main(page: ft.Page):
        page.title = "Chord Detector"
        page.window_width = 300
        page.window_height = 200

        chord_text = ft.Text("Waiting for chord...", size=20)
        page.add(chord_text)

        def update_chord():
            while True:
                chord = chord_queue.get()
                if chord:
                    chord_text.value = f"Chord: {chord}"
                else:
                    chord_text.value = f"Chord: ___"
                page.update()

        threading.Thread(target=update_chord, daemon=True).start()

    ft.app(target=main)


def process_audio(chord_queue, interpreter, le, input_details, output_index, key, samples_per_buffer): # only for the continuous mode
    audio_buffer = np.array([], dtype=np.float32)
    while True:
        while len(audio_buffer) < samples_per_buffer:
            data = audio_queue.get()
            audio_buffer = np.concatenate([audio_buffer, data.flatten()])

        chunk = audio_buffer[:samples_per_buffer]
        audio_buffer = audio_buffer[samples_per_buffer:]
        transposed_audio, semitones = transpose_audio(chunk, key)
        predicted_chord_in_c = predict_class(False, transposed_audio, interpreter, le, input_details, output_index)

        if predicted_chord_in_c is None:
            print("No chord detected")
        else:
            print(f"\nModel Predicted chord (in key of C): {predicted_chord_in_c}")
            true_chord = transpose_back_chord(predicted_chord_in_c, semitones)
            chord_queue.put(true_chord)
            print(f"True Transposed Chord (in original key): {true_chord}")


def main():
    interpreter = tf.lite.Interpreter(model_path="/Users/jonathanyu/Desktop/coding/chord identification/.venv/model.tflite")
    interpreter.allocate_tensors()  # model path differs on raspberry pi
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']
    classes = ['A minor', 'Bb major', 'B diminished', 'C major', 'D minor', 'E minor', 'F major', 'G major']
    le = LabelEncoder()
    le.fit(classes)


    learn = False
    mode = get_mode()
    if mode.lower() == "continuous":
        buffer_seconds = 1
    elif mode.lower() == "learn":
        buffer_seconds = 4
        learn = True
    else:
        print("Mode not valid")
        exit()

    key = get_key()
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=1024, latency='high')
    print("Starting audio stream...")
    stream.start()

    samples_per_buffer = int(sample_rate * buffer_seconds)

    if learn:
        engine.say("Starting...")
        engine.runAndWait()
        print("Please play the chord")
        engine.say("Please play the chord")
        engine.runAndWait()
        while True:
            audio_buffer = np.array([], dtype=np.float32)
            while len(audio_buffer) < samples_per_buffer:
                data = audio_queue.get()
                audio_buffer = np.concatenate([audio_buffer, data.flatten()])
            chunk = audio_buffer[:samples_per_buffer]
            transposed_audio, semitones = transpose_audio(chunk, key)
            predicted_chord_in_c = predict_class(learn, transposed_audio, interpreter, le, input_details, output_index)
            if predicted_chord_in_c is None:
                print("No chord detected, please play the chord")
                engine.say("No chord detected, please play the chord")
                engine.runAndWait()
            else:
                print(f"\nModel Predicted chord (in key of C): {predicted_chord_in_c}")
                true_chord = transpose_back_chord(predicted_chord_in_c, semitones)
                print(f"True Transposed Chord (in original key): {true_chord}")
                engine.say(true_chord)
                engine.runAndWait()
                print("Waiting...")
                engine.say("Waiting")
                engine.runAndWait()
                stream.stop()
                time.sleep(3)
                print("Please play the next chord")
                engine.say("Please play the next chord")
                engine.runAndWait()
                time.sleep(0.1)
                stream.start()
    else:
        chord_queue = queue.Queue()
        audio_thread = threading.Thread(
            target=process_audio,
            args=(chord_queue, interpreter, le, input_details, output_index, key, samples_per_buffer),
            daemon=True
        )
        audio_thread.start()
        run_flet(chord_queue)


if __name__ == '__main__':
    main()
