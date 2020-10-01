# IMPORTANT NOTE
**I'm not working on this project anymore**.
I advise everyone curious about voice detection to have a look at some more modern approaches using deep learning, like:
 - https://www.mathworks.com/help/audio/examples/voice-activity-detection-in-noise-using-deep-learning.html
 - https://medium.com/vivolab/vivovad-a-voice-activity-detection-tool-based-on-recurrent-neural-networks-32356526321c
 - https://github.com/hcmlab/vadnet
# Voice Activity Detector
Python code to apply voice activity detector to wave file.
Voice activity detector based on ration between energy in speech band and total energy.

## Requirements

* numpy
* scipy
* matplotlib
* tkinter (sudo apt install python3-tk)

## Basic Idea
Input audio data treated as following:

1. Convert stereo to mono.
2. Move a window of 20ms along the audio data.
3. Calculate the ratio between energy of speech band and total energy for window.
4. If ratio is more than threshold (0.6 by default) label windows as speech.
5. Apply median filter with length of 0.5s to smooth detected speech regions.
6. Represent speech regions as intervals of time.

## How To
Create object:

1. import vad module.
2. create instance of class VoiceActivityDetector with full path to wave file.
3. run method to detect speech regions.
4. optionally, plot original wave data and detected speech region.

Example python script which saves speech intervals in json file:

`./detectVoiceInWave.py ./wav-sample.wav ./results.json`

Example python code to plot detected speech regions:
```python
from vad import VoiceActivityDetector

filename = '/Users/user/wav-sample.wav'
v = VoiceActivityDetector(filename)
v.plot_detected_speech_regions()
```

-------
Alexander USOLTSEV 2015 (c) [MIT License](https://opensource.org/licenses/MIT)
