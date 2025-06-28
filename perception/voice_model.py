# voice_model.py model integration placeholder
# voice_model.py
import speech_recognition as sr

class VoiceModel:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
    
    def listen(self, timeout=5, phrase_time_limit=10):
        with self.microphone as source:
            print("Listening for voice input...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return audio

    def recognize_speech(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None