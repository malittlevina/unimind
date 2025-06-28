

import pyaudio
import wave

class MicListener:
    def __init__(self, rate=44100, chunk=1024, channels=1, format=pyaudio.paInt16):
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.format = format
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_stream(self):
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        print("Microphone stream started.")

    def read_audio(self):
        if self.stream:
            data = self.stream.read(self.chunk)
            return data
        else:
            print("Stream is not active.")
            return None

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            print("Microphone stream stopped.")

if __name__ == "__main__":
    listener = MicListener()
    try:
        listener.start_stream()
        for _ in range(100):  # Read 100 chunks then stop
            audio_data = listener.read_audio()
            if audio_data:
                print("Captured audio chunk.")
    finally:
        listener.stop_stream()