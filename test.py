import sounddevice as sd
import numpy as np
from pvrecorder import PvRecorder

def list_devices():
    print("Available input devices:")
    for index, name in enumerate(PvRecorder.get_available_devices()):
        print(f"[{index}] {name}")

def record_and_play(device_index):
    duration = 3  # seconds
    samplerate = 16000
    print(f"Recording from device {device_index} for {duration} seconds...")
    
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_index)
    sd.wait()
    
    print("Playback...")
    sd.play(recording, samplerate)
    sd.wait()

if __name__ == "__main__":
    list_devices()
    try:
        choice = int(input("Enter the device index you want to test: "))
        record_and_play(choice)
    except ValueError:
        print("Invalid input. Please enter a number.")
