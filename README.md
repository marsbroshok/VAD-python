# Voice Activity Detector
Python code to apply voice activity detector to wave file.
Voice activity detector based on ration between energy in speech band and total energy.

## Requirements

* numpy
* scipy
* matplotlib

## Basic Idea
Input audio data treated as following:

1. Convert stereo to mono
2. Move a window of 20ms along the audio data
3. Calculate ration between energy of speech band and total energy for window
4. If ratio is more than threshold (0.6 by default) label windows as speech
5. Apply median filter with length of 0.5s to smooth detected speech regions
6. Represent speech regions as intervals of time

## How To
Create object:

1. import vad module
2. create instance of class VoiceActivityDetector with full path to wave file
3. run method to detect speech regions
4. optionally, plot original wave data and detected speech region

Example python script which saves speech intervals in json file:

`./detectVoiceInWave.py ./wav-sample.wav ./results.json`

Example pyhton code to plot detected speech regions:
```python
from vad import VoiceActivityDetector

filename = '/Users/user/wav-sample.wav'
v = VoiceActivityDetector(filename)
v.plot_detected_speech_regions()
```
