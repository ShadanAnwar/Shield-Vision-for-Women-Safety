import streamlit as st
import tempfile
import os
from video_model import VideoModel
from audio_analyzer import AudioAnalyzer
from alert_system import AlertSystem
from logger import Logger
import time

class ShieldVisionApp:
    def __init__(self):
        st.set_page_config(page_title="Shield Vision", page_icon="🛡️", layout="wide")
        self.logger = Logger()
        self.video_model = VideoModel(model_path="testing UI/MoBiLSTM_best_model.keras")
        self.audio_analyzer = AudioAnalyzer()
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self.alert_system = AlertSystem(
            api_type="email",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD")
        )
        if 'emergency_contact' not in st.session_state:
            st.session_state.emergency_contact = self.logger.get_emergency_contact()

    @st.cache_resource
    def load_models(self):
        return self.video_model, self.audio_analyzer

    def run(self):
        self._render_header()
        self._render_settings()
        tab1, tab2 = st.tabs(["Video Analysis", "Audio Analysis"])
        with tab1:
            self._render_video_tab()
        with tab2:
            self._render_audio_tab()

    def _render_header(self):
        st.title("Shield Vision")
        st.markdown("Women's Safety System")

    def _render_settings(self):
        with st.sidebar:
            st.header("Settings")
            contact = st.text_input(
                "Emergency Contact Email",
                value=st.session_state.emergency_contact,
                placeholder="your.email@example.com",
                help="Please provide a valid email address for alerts"
            )
            if st.button("Update Contact"):
                # Use the same email validation as in AlertSystem
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if re.match(email_pattern, contact):
                    self.logger.update_emergency_contact(contact)
                    st.session_state.emergency_contact = contact
                    st.success("Contact updated!")
                else:
                    st.error("Please enter a valid email address")
                    # Log the validation error
                    self.logger.log_incident("ERROR: validation", "medium", f"Invalid email address: {contact}")

    def _render_video_tab(self):
        st.header("Video Analysis")
        uploaded_file = st.file_uploader("Upload Video (MP4/AVI)", type=["mp4", "avi"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            st.video(uploaded_file)
            if st.button("Analyze Video"):
                with st.spinner("Analyzing video..."):
                    prediction, confidence = self.video_model.predict(temp_path)
                    self._display_video_results(prediction, confidence, uploaded_file.name)
            os.unlink(temp_path)

    def _display_video_results(self, prediction, confidence, filename):
        st.subheader("Results")
        if prediction == "Violence":
            st.error("⚠️ Threat Detected!")
            severity = "high"
            self.logger.log_incident("Video-Violence", severity, f"File: {filename}, Confidence: {confidence:.2f}")
            self._send_alert("video", filename)
        else:
            st.success("✅ All Clear")
            severity = "low"
            self.logger.log_incident("Video-NonViolence", severity, f"File: {filename}, Confidence: {confidence:.2f}")
        st.metric("Prediction", prediction)
        st.metric("Confidence", f"{confidence:.2f}")

    def _render_audio_tab(self):
        st.header("Audio Analysis")
        uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
        if uploaded_file:
            # Use the correct file extension based on the uploaded file
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if not file_ext:
                file_ext = '.wav' if uploaded_file.type == 'audio/wav' else '.mp3'

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            st.audio(uploaded_file)

            if st.button("Analyze Audio"):
                with st.spinner("Analyzing audio..."):
                    try:
                        results = self.audio_analyzer.analyze_audio(temp_path)
                        self._display_audio_results(results, uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error analyzing audio: {str(e)}")
                        self.logger.log_incident("ERROR: audio_analysis", "high", str(e))

            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    def _display_audio_results(self, results, filename):
        st.subheader("Audio Analysis Results")

        # Display technical analysis
        with st.expander("Technical Details"):
            st.write(f"Volume Level: {'Shouting' if results['audio_properties']['is_shouting'] else 'Normal'}")
            st.write(f"Pitch: {'High' if results['audio_properties']['is_high_pitch'] else 'Normal'}")
            st.write(f"Dominant Frequency: {results['audio_properties']['dominant_freq']:.2f} Hz")

        # Display threat analysis
        if results["is_threat"]:
            st.error(f"⚠️ Threat Detected: {results['threat_type'].upper()}")
            st.warning(f"Confidence: {results['confidence']}%")
            st.info(f"Reason: {results['reason']}")

            self.logger.log_incident(
                f"Audio-{results['threat_type'].capitalize()}",
                "high" if results['threat_type'] == "violence" else "medium",
                {
                    "file": filename,
                    "confidence": results["confidence"],
                    "reason": results["reason"]
                }
            )
            # Extract only the needed fields for the alert
            alert_details = {
                "threat_type": results["threat_type"],
                "confidence": results["confidence"],
                "reason": results["reason"]
            }
            self._send_alert("audio", filename, alert_details)
        else:
            st.success("✅ No Threat Detected")
            if results['audio_properties']['is_shouting']:
                st.info("Note: Shouting was detected but determined to be non-threatening")

            self.logger.log_incident(
                "Audio-Normal",
                "low",
                {
                    "file": filename,
                    "analysis": results["reason"]
                }
            )

    def _format_alert_message(self, detection_type, details=None):
        """Generate precise alert messages"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"{detection_type.capitalize()} Threat Alert\n"
        message += f"Type: {details.get('threat_type', 'unknown').upper()}\n"
        message += f"Confidence: {details.get('confidence', 0)}%\n"
        message += f"Analysis: {details.get('reason', 'No details')}\n"
        message += f"Timestamp: {timestamp}"
        return message

    def _send_alert(self, detection_type, filename, details=None):
        if st.session_state.emergency_contact:
            # Prepare details based on detection type
            if detection_type == "video":
                alert_details = {"threat_type": "violence", "confidence": 95, "reason": f"Violence detected in video: {filename}"}
            elif detection_type == "audio" and isinstance(details, dict):
                alert_details = details
            else:
                alert_details = {"threat_type": "unknown", "confidence": 0, "reason": "No details available"}

            # Format and send the alert
            alert_msg = self._format_alert_message(detection_type, alert_details)
            success, message = self.alert_system.send_alert(alert_msg, st.session_state.emergency_contact)

            # Display feedback to the user
            if success:
                st.success(f"Alert sent to {st.session_state.emergency_contact}")
            else:
                st.error(f"Failed to send alert: {message}")
                # Log the error
                self.logger.log_incident("ERROR: alert", "high", message)


if __name__ == "__main__":
    app = ShieldVisionApp()
    app.run()