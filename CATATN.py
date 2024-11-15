import speech_recognition as sr
import pyttsx3
import cv2
import subprocess
import random
import os
import time
import numpy as np
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class CATATN:
    def __init__(self):
        print("Initializing CATATN AI System...")
        self.initialize_models()
        self.initialize_voice()
        self.initialize_personality()
        
    def initialize_models(self):
        """Initialize all AI models and components"""
        # Voice Recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        
        # Face Detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Text Generation Model
        try:
            self.conversation_model = pipeline("text-generation", model="gpt2")
        except Exception as e:
            print(f"Warning: Could not load GPT-2 model. Using basic responses instead. Error: {e}")
            self.conversation_model = None

    def initialize_voice(self):
        """Initialize text-to-speech with multiple voices"""
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.available_voices = voices
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)

    def initialize_personality(self):
        """Initialize personality system"""
        self.personality = {
            'current_mode': 'friendly',
            'voices': {
                'professional': {'rate': 150},
                'friendly': {'rate': 170},
                'energetic': {'rate': 190}
            }
        }

    def speak(self, text, mode='friendly'):
        """Enhanced speaking with different modes"""
        try:
            # Set voice properties based on mode
            voice_settings = self.personality['voices'].get(mode, self.personality['voices']['friendly'])
            self.engine.setProperty('rate', voice_settings['rate'])
            
            print(f"CATATN: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")
            print(f"CATATN: {text}")

    def listen(self):
        """Enhanced listening with noise reduction"""
        with sr.Microphone() as source:
            print("\nListening...")
            try:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                self.speak("I couldn't understand that. Could you please repeat?")
                return None
            except sr.RequestError:
                self.speak("I'm having trouble with my speech recognition service.")
                return None
            except Exception as e:
                print(f"Listening Error: {e}")
                return None

    def generate_response(self, user_input):
        """Generate responses to user input"""
        try:
            if self.conversation_model:
                # Generate response using GPT-2
                prompt = f"User: {user_input}\nAssistant:"
                response = self.conversation_model(prompt, 
                                                max_length=50,
                                                num_return_sequences=1,
                                                temperature=0.7)[0]['generated_text']
                
                # Extract only the assistant's response
                response = response.split("Assistant:")[-1].strip()
                return response
            else:
                # Fallback to basic responses
                responses = [
                    "I understand. Please tell me more.",
                    "That's interesting! How can I help?",
                    "I'm here to assist you.",
                    "Could you elaborate on that?",
                    "I'm processing what you said. What would you like me to do?"
                ]
                return random.choice(responses)
        except Exception as e:
            print(f"Response Generation Error: {e}")
            return "I'm here to help. What would you like me to do?"

    def detect_faces(self):
        """Basic face detection using webcam"""
        try:
            cap = cv2.VideoCapture(0)
            print("Press 'q' to quit face detection")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    self.speak("I see a face!")
                
                cv2.imshow('CATATN Vision', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Face Detection Error: {e}")
            self.speak("I'm having trouble with the camera.")

def main():
    """Main function to run CATATN"""
    catatn = CATATN()
    
    # Welcome message
    catatn.speak("Hello! I am CATATN, your AI assistant. I can help you with:")
    catatn.speak("1. Having conversations")
    catatn.speak("2. Face detection")
    catatn.speak("How can I assist you today?")
    
    while True:
        user_input = catatn.listen()
        
        if user_input:
            if 'quit' in user_input.lower() or 'exit' in user_input.lower():
                catatn.speak("Goodbye! Have a great day!")
                break
            
            if 'face' in user_input.lower() or 'detect' in user_input.lower():
                catatn.speak("Starting face detection. Press 'q' to stop.")
                catatn.detect_faces()
            else:
                response = catatn.generate_response(user_input)
                catatn.speak(response)

if __name__ == "__main__":
    main()
