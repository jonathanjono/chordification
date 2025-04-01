# Chord Detection and Transposition System

## Overview
This project is a **real-time chord detection system** that uses:
- **Speech recognition** for user input.
- **Audio processing** with Mel spectrograms derrived from the original training script.
- **A machine learning model** to classify and transpose musical chords.
- **A simple graphical user interface (GUI)** built with **Flet**.

It allows musicians to **detect chords in real-time**, **transpose them to a different key**, and **display them in a simple interface**, imitating **human's relative pitch system**.

## 📂 Dataset and Training Script
The dataset and training script is from Kaggle:  
[GUITAR CHORDS V3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3)
[Guitar Chord Classification (95% Accuracy)](https://www.kaggle.com/code/akshaysom/guitar-chord-classification-test-accuracy-95)

---

## ✨ Features
✅ **Speech Recognition** – Users can **speak** the mode and key they want.  
✅ **Real-time Audio Processing** – Captures and processes live audio input.  
✅ **Machine Learning Model** – Predicts chords with **TensorFlow Lite**.  
✅ **Automatic Transposition** – Adjusts detected chords to match the **selected key**.  
✅ **User-Friendly GUI** – Displays detected chords with **Flet**.  

### Transposition System
**Mapping Keys**: 
- The code uses a dictionary that assigns a specific semitone shift to each key 
- For example, key of C has 0 shift, D has -2, B would have a +1 semitone shift because it shifts the note the closest way to C to make the shift as small as possible
- Makes it so that all chords are evaluated as if they were in the key of C
**Applying Pitch Shift**: Using the librosa.effects.pitch_shift function, the program transposes the audio by the number of semitones correlated with each note T
**Reversing the Shift**: After the model predicts the chord in the transposed (key of C) space, the program reverses the shift to display the chord in the original key\

---

## 🔧 Installation
Dependencies:

```sh
pip install numpy pandas librosa tensorflow sounddevice speechrecognition pyttsx3 flet scipy scikit-learn
