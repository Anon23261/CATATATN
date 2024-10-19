import speech_recognition as sr
import pyttsx3
import cv2
import subprocess
import random
import os
import time
from transformers import pipeline
import tensorflow as tf

# -------------------------------
# CATATN Initialization
# -------------------------------

class CATATN:
    def __init__(self):
        # Initial personality and learning models
        self.personality = "neutral"
        self.voices = ["calm", "serious", "humorous", "empathetic"]
        self.engine = pyttsx3.init()
        self.code_model = pipeline("text-generation", model="gpt-2")
        self.recognizer = sr.Recognizer()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # -------------------------------
    # 1. Voice Interaction System
    # -------------------------------
    def speak(self, text):
        """Speak out the given text."""
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Listen for voice input and convert to text."""
        with sr.Microphone() as source:
            print("CATATN is listening...")
            audio = self.recognizer.listen(source)
            try:
                user_input = self.recognizer.recognize_google(audio)
                print(f"You said: {user_input}")
                return user_input
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't catch that.")
                return None

    # -------------------------------
    # 2. Personality System
    # -------------------------------
    def set_personality(self, mode):
        """Set personality mode for CATATN."""
        if mode in self.voices:
            self.personality = mode
        else:
            self.speak("Invalid personality mode. Setting to neutral.")
            self.personality = "neutral"

    def respond(self):
        """Generate responses based on personality."""
        responses = {
            "calm": ["I see, let's handle this together.", "Stay cool. Everything is under control."],
            "serious": ["Let's stay focused and solve this.", "No distractions. We have work to do."],
            "humorous": ["Why don’t programmers like nature? It has too many bugs!", "Let’s hack this with a side of fun."],
            "empathetic": ["I understand, I’m here for you.", "Tell me more, I’m listening."],
            "neutral": ["I'm here to help.", "Let's proceed."]
        }
        return random.choice(responses.get(self.personality, responses["neutral"]))

    # -------------------------------
    # 3. Facial Recognition System
    # -------------------------------
    def detect_emotions(self, frame):
        """Detect emotions from the frame using a pre-trained model."""
        # Placeholder for emotion detection logic
        # emotion_model = load_emotion_model()  # Load your emotion detection model
        # emotions = emotion_model.predict(frame)  # Predict emotions
        return "happy"  # Example emotion

    def detect_faces(self):
        """Detect faces and recognize emotions using webcam."""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                emotion = self.detect_emotions(frame[y:y+h, x:x+w])
                self.speak(f"You look {emotion}!")  # Respond based on detected emotion
            cv2.imshow('CATATN - Facial Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------
    # 4. Cybersecurity Specialist
    # -------------------------------
    def run_nmap(self, target_ip):
        """Run an Nmap scan on the target IP."""
        self.speak(f"Scanning network for target: {target_ip}")
        command = f"nmap -sS {target_ip}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if output:
            self.speak(f"Nmap scan complete for {target_ip}")
            return output.decode('utf-8')
        else:
            self.speak(f"Error in scanning {target_ip}")
            return None

    def metasploit(self, exploit):
        """Run Metasploit to exploit a vulnerability."""
        self.speak(f"Attempting to exploit vulnerability: {exploit}")
        # Placeholder for actual Metasploit execution
        # Example command:
        # subprocess.call(["msfconsole", "-q", "-x", f"use {exploit}; exploit; exit"])
        return f"Exploiting vulnerability with {exploit}"

    # -------------------------------
    # 5. Programming Expertise
    # -------------------------------
    def generate_code(self, prompt):
        """Use a language model to generate code based on a prompt."""
        self.speak("Let me generate some code for you.")
        result = self.code_model(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        return result

    # -------------------------------
    # 6. Machine Learning Expansion
    # -------------------------------
    def learn_from_data(self, training_data, labels):
        """Train a simple neural network on provided data."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(training_data, labels, epochs=5)
        self.speak("Learning complete. CATATN has expanded its knowledge.")

# -------------------------------
# Running CATATN
# -------------------------------

if __name__ == "__main__":
    # Initialize CATATN
    catatn = CATATN()

    # Example of personality setting and response
    catatn.set_personality("humorous")
    response = catatn.respond()
    catatn.speak(response)

    # Example of listening and responding
    user_input = catatn.listen()
    if user_input:
        catatn.speak(f"You said: {user_input}")

    # Example of facial recognition (commented out for non-webcam systems)
    # catatn.detect_faces()

    # Example of running an Nmap scan
    target_ip = "192.168.1.1"
    scan_result = catatn.run_nmap(target_ip)
    print(scan_result)

    # Example of generating code
    code_prompt = "Write a Python function to reverse a string"
    generated_code = catatn.generate_code(code_prompt)
    print(generated_code)

    # Example of training the neural network (with dummy data)
    # dummy_data = [[0]*64] * 100
    # dummy_labels = [0] * 100
    # catatn.learn_from_data(dummy_data, dummy_labels)
