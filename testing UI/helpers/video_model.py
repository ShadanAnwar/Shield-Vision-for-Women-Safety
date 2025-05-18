import os
import cv2
import numpy as np
import tensorflow as tf

class VideoModel:
    def __init__(self, model_path=None):
        self.image_height = 64
        self.image_width = 64
        self.sequence_length = 16
        self.class_names = ["NonViolence", "Violence"]
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Model path {model_path} not found. Using placeholder model.")
            self._create_placeholder_model()

    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def _create_placeholder_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.sequence_length, self.image_height, self.image_width, 3)),
            tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu'),
            tf.keras.layers.MaxPooling3D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model = model

    def extract_frames(self, video_path):
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / self.sequence_length), 1)
        for frame_counter in range(self.sequence_length):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (self.image_height, self.image_width))
            normalized_frame = resized_frame / 255.0
            frames_list.append(normalized_frame)
        video_reader.release()
        if len(frames_list) < self.sequence_length:
            padded_frames = np.zeros((self.sequence_length, self.image_height, self.image_width, 3))
            padded_frames[:len(frames_list)] = frames_list
            return padded_frames
        return np.array(frames_list)

    def predict(self, video_path):
        if not self.model:
            return "Error", 0.0
        frames = self.extract_frames(video_path)
        frames = np.expand_dims(frames, axis=0)
        prediction = self.model.predict(frames, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])
        return self.class_names[class_idx], confidence