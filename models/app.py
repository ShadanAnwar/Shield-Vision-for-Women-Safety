import streamlit as st
import tempfile
import os
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf
import sqlite3
import json
import requests
from io import BytesIO
import time

# Set page configuration
st.set_page_config(
    page_title="Shield Vision",
    page_icon="🛡️",
    layout="wide"
)

# Database setup
def setup_db():
    conn = sqlite3.connect('shield_vision.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS incidents
    (id INTEGER PRIMARY KEY, 
    timestamp TEXT, 
    type TEXT, 
    severity TEXT,
    details TEXT)
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS settings
    (id INTEGER PRIMARY KEY, 
    emergency_contact TEXT,
    alert_enabled INTEGER)
    ''')
    
    # Check if settings exist, if not create default
    c.execute("SELECT COUNT(*) FROM settings")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO settings (emergency_contact, alert_enabled) VALUES (?, ?)", 
                 ('', 1))
    
    conn.commit()
    conn.close()

def log_incident(incident_type, severity, details):
    conn = sqlite3.connect('shield_vision.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO incidents (timestamp, type, severity, details) VALUES (?, ?, ?, ?)",
             (timestamp, incident_type, severity, details))
    conn.commit()
    conn.close()

def get_emergency_contact():
    conn = sqlite3.connect('shield_vision.db')
    c = conn.cursor()
    c.execute("SELECT emergency_contact FROM settings WHERE id = 1")
    contact = c.fetchone()[0]
    conn.close()
    return contact

def update_emergency_contact(phone):
    conn = sqlite3.connect('shield_vision.db')
    c = conn.cursor()
    c.execute("UPDATE settings SET emergency_contact = ? WHERE id = 1", (phone,))
    conn.commit()
    conn.close()

def is_alert_enabled():
    conn = sqlite3.connect('shield_vision.db')
    c = conn.cursor()
    c.execute("SELECT alert_enabled FROM settings WHERE id = 1")
    enabled = c.fetchone()[0]
    conn.close()
    return enabled == 1

def toggle_alert_system(enabled):
    conn = sqlite3.connect('shield_vision.db')
    c = conn.cursor()
    c.execute("UPDATE settings SET alert_enabled = ? WHERE id = 1", (1 if enabled else 0,))
    conn.commit()
    conn.close()

# Initialize the database
setup_db()

# Load the video analysis model
@st.cache_resource
def load_video_model():
    try:
        # Placeholder for actual model loading
        # In real implementation, you would load the saved model
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(16, 64, 64, 3)),
            tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu'),
            tf.keras.layers.MaxPooling3D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Note: In production, replace this with 
        # model = tf.keras.models.load_model('path_to_your_model')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Video processing functions
def extract_frames(video_path, sequence_length=16, image_height=64, image_width=64):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    
    # Get frame count
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/sequence_length), 1)
    
    # Extract frames at regular intervals
    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        
        if not success:
            break
            
        # Resize and normalize frame
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    
    # Ensure we have the right number of frames
    if len(frames_list) == sequence_length:
        return np.array(frames_list)
    else:
        # If not enough frames, pad with zeros
        padded_frames = np.zeros((sequence_length, image_height, image_width, 3))
        padded_frames[:len(frames_list)] = np.array(frames_list)
        return padded_frames

def predict_video(video_path, model):
    # Extract frames
    frames = extract_frames(video_path)
    
    if frames is None:
        return "Error", 0.0
    
    # Make prediction
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)[0]
    
    # Get class with highest probability
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    
    # Map index to class name
    class_names = ["NonViolence", "Violence"]
    return class_names[class_idx], float(confidence)

# Audio processing functions
def transcribe_audio(audio_file):
    # In a real implementation, you would call Whisper API or use a local model
    # For this demo, we'll simulate transcription
    
    # Simulate processing time
    time.sleep(2)
    
    # Generate simulated transcription based on file name
    filename = os.path.basename(audio_file)
    if "help" in filename.lower() or "emergency" in filename.lower():
        return "Help! I need help! Someone is following me!", True
    elif "threat" in filename.lower():
        return "Stop following me! Leave me alone! I'm calling the police!", True
    else:
        return "Hey, I'm on my way home. Should be there in about 20 minutes.", False

def analyze_audio_content(transcript):
    # Simple rule-based classification
    threat_keywords = ['help', 'emergency', 'police', 'stop', 'following', 
                       'leave me alone', 'scared', 'afraid', 'threatening',
                       'harassing', 'assault', 'attack', 'weapon', 'knife', 'gun']
    
    transcript_lower = transcript.lower()
    detected_keywords = [word for word in threat_keywords if word in transcript_lower]
    
    if detected_keywords:
        return "Threat Detected", detected_keywords, "high" if len(detected_keywords) > 2 else "medium"
    else:
        return "No Threat", [], "none"

# Alert system functions
def send_alert(message, phone_number):
    # In production, implement actual Twilio or Textbelt integration
    # For now, we'll simulate sending an alert
    
    st.warning("ALERT SYSTEM: Simulating sending alert to " + phone_number)
    
    # Simulate API call delay
    time.sleep(1)
    
    # Return success for demo
    return True, "Alert sent successfully"

# UI Components
def render_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://via.placeholder.com/150x150.png?text=SV", width=80)
    with col2:
        st.title("Shield Vision")
        st.markdown("##### Women's Safety System")

def render_settings_sidebar():
    with st.sidebar:
        st.header("Settings")
        
        # Emergency Contact
        current_contact = get_emergency_contact()
        new_contact = st.text_input("Emergency Contact Number", 
                                   value=current_contact,
                                   placeholder="+1234567890")
        
        # Save button for contact
        if st.button("Update Contact"):
            update_emergency_contact(new_contact)
            st.success("Emergency contact updated!")
        
        # Alert toggle
        alert_enabled = is_alert_enabled()
        new_alert_state = st.checkbox("Enable Alert System", value=alert_enabled)
        
        # Update alert state if changed
        if new_alert_state != alert_enabled:
            toggle_alert_system(new_alert_state)
            status = "enabled" if new_alert_state else "disabled"
            st.info(f"Alert system {status}")
        
        # Recent incidents
        st.header("Recent Incidents")
        
        # Connect to db and fetch recent incidents
        conn = sqlite3.connect('shield_vision.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, type, severity FROM incidents ORDER BY timestamp DESC LIMIT 5")
        incidents = c.fetchall()
        conn.close()
        
        if incidents:
            for idx, (time, type, severity) in enumerate(incidents):
                severity_color = "red" if severity == "high" else "orange" if severity == "medium" else "green"
                st.markdown(f"{idx+1}. **{time}** - {type} (*{severity}*)")
        else:
            st.info("No recent incidents recorded")

def render_video_analysis_tab():
    st.header("Video Analysis")
    
    # Upload video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Display video
        st.video(uploaded_file)
        
        # Analyze button
        if st.button("Analyze Video"):
            with st.spinner("Analyzing video..."):
                # Get model and make prediction
                model = load_video_model()
                if model:
                    prediction, confidence = predict_video(temp_path, model)
                    
                    # Display result
                    st.subheader("Analysis Results")
                    
                    # Status indicator
                    if prediction == "Violence":
                        st.error("⚠️ THREAT DETECTED ⚠️")
                        severity = "high"
                        
                        # Log incident
                        log_incident("Video-Violence", severity, 
                                    f"Detected in video: {uploaded_file.name}, Confidence: {confidence:.2f}")
                        
                        # Send alert if enabled
                        if is_alert_enabled():
                            contact = get_emergency_contact()
                            if contact:
                                alert_msg = f"ALERT: Violence detected in video analysis. Location data unavailable. This is an automated message from Shield Vision."
                                success, message = send_alert(alert_msg, contact)
                                if success:
                                    st.success("Emergency alert sent")
                                else:
                                    st.error(f"Failed to send alert: {message}")
                            else:
                                st.warning("No emergency contact set. Alert not sent.")
                    else:
                        st.success("✅ NO THREAT DETECTED")
                    
                    # Prediction details
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric("Confidence", f"{confidence:.2f}")
                    
                    # Add some space
                    st.write("")
                    
                else:
                    st.error("Failed to load analysis model")
        
        # Clean up the temp file
        try:
            os.unlink(temp_path)
        except:
            pass

def render_audio_analysis_tab():
    st.header("Audio Analysis")
    
    # Upload audio
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Display audio player
        st.audio(uploaded_file)
        
        # Analyze button
        if st.button("Analyze Audio"):
            with st.spinner("Transcribing and analyzing audio..."):
                # Transcribe audio
                transcript, emergency_words_detected = transcribe_audio(temp_path)
                
                # Analyze transcript
                result, keywords, severity = analyze_audio_content(transcript)
                
                # Display transcript
                st.subheader("Transcription")
                st.write(transcript)
                
                # Display analysis results
                st.subheader("Analysis Results")
                
                # Status indicator
                if result == "Threat Detected":
                    st.error("⚠️ THREAT DETECTED ⚠️")
                    
                    # Log incident
                    log_incident("Audio-Threat", severity, 
                                f"Detected in audio: {uploaded_file.name}, Keywords: {', '.join(keywords)}")
                    
                    # Send alert if enabled
                    if is_alert_enabled():
                        contact = get_emergency_contact()
                        if contact:
                            alert_msg = f"ALERT: Potential threat detected in audio. Keywords: {', '.join(keywords)}. This is an automated message from Shield Vision."
                            success, message = send_alert(alert_msg, contact)
                            if success:
                                st.success("Emergency alert sent")
                            else:
                                st.error(f"Failed to send alert: {message}")
                        else:
                            st.warning("No emergency contact set. Alert not sent.")
                else:
                    st.success("✅ NO THREAT DETECTED")
                
                # Display detected keywords if any
                if keywords:
                    st.write("Detected keywords:")
                    st.write(", ".join(keywords))
        
        # Clean up the temp file
        try:
            os.unlink(temp_path)
        except:
            pass

# Main App Layout
def main():
    # Render header
    render_header()
    
    # Render settings sidebar
    render_settings_sidebar()
    
    # Tabs for different analysis types
    tab1, tab2 = st.tabs(["Video Analysis", "Audio Analysis"])
    
    with tab1:
        render_video_analysis_tab()
    
    with tab2:
        render_audio_analysis_tab()

# Run the app
if __name__ == "__main__":
    main()
