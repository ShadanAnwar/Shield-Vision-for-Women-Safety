import streamlit as st
import tempfile
import os

from helpers.video_model import VideoModel
from helpers.audio_analyzer import AudioAnalyzer
from helpers.alert_mail_system import AlertMailSystem
from helpers.alert_sms_system import AlertSmsSystem
from helpers.logger import Logger
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
        self.alert_mail_system = AlertMailSystem(
            api_type="email",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD")
        )
        self.alert_sms_system = AlertSmsSystem(api_type="sms")
        if 'emergency_contact' not in st.session_state:
            st.session_state.emergency_contact = {
                'email': self.logger.get_emergency_contact('email'),
                'phone': self.logger.get_emergency_contact('phone')
            }

    @st.cache_resource
    def load_models(self):
        return self.video_model, self.audio_analyzer

    def run(self):
        self._render_header()
        self._render_settings()
        tab1, tab2, tab3 = st.tabs(["Video Analysis", "Audio Analysis", "Incident Log"])
        with tab1:
            self._render_video_tab()
        with tab2:
            self._render_audio_tab()
        with tab3:
            self._render_log_tab()

    def _render_header(self):
        st.title("Shield Vision")
        st.markdown("Women's Safety System")

    def _render_settings(self):
        with st.sidebar:
            st.header("Settings")
            st.subheader("Emergency Contact")
            email = st.text_input(
                "Emergency Contact Email",
                value=st.session_state.emergency_contact['email'],
                placeholder="your.email@example.com",
                help="Please provide a valid email address for alerts"
            )
            st.write(f"Current Receiver Email: {st.session_state.emergency_contact['email']}")
            phone = st.text_input(
                "Emergency Contact Phone",
                value=st.session_state.emergency_contact['phone'] or "+91",
                placeholder="+919876543210",
                help="Enter a 10-digit phone number with +91 prefix"
            )
            if st.button("Update Contact"):
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                phone_clean = phone.replace('+91', '') if phone.startswith('+91') else phone
                phone_pattern = r'^\d{10}$'
                valid = True
                if email and not re.match(email_pattern, email):
                    st.error("Please enter a valid email address")
                    self.logger.log_incident("ERROR: validation", "medium", f"Invalid email address: {email}")
                    valid = False
                if phone_clean and not re.match(phone_pattern, phone_clean):
                    st.error("Please enter a valid 10-digit phone number")
                    self.logger.log_incident("ERROR: validation", "medium", f"Invalid phone number: {phone}")
                    valid = False
                if valid:
                    if email:
                        self.logger.update_emergency_contact(email, 'email')
                        st.session_state.emergency_contact['email'] = email
                    if phone_clean:
                        phone = f"+91{phone_clean}"
                        self.logger.update_emergency_contact(phone, 'phone')
                        st.session_state.emergency_contact['phone'] = phone
                    st.success("Contact updated!")

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
            try:
                os.unlink(temp_path)
            except:
                pass

    def _display_audio_results(self, results, filename):
        st.subheader("Audio Analysis Results")
        with st.expander("Technical Details"):
            st.write(f"Volume Level: {'Shouting' if results['audio_properties']['is_shouting'] else 'Normal'}")
            st.write(f"Pitch: {'High' if results['audio_properties']['is_high_pitch'] else 'Normal'}")
            st.write(f"Dominant Frequency: {results['audio_properties']['dominant_freq']:.2f} Hz")
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

    def _render_log_tab(self):
        st.header("Incident Log")
        incidents = self.logger.get_recent_incidents(limit=10)
        if incidents:
            st.subheader("Recent Incidents")
            for timestamp, incident_type, severity, details in incidents:
                with st.expander(f"{timestamp} - {incident_type}"):
                    st.write(f"**Severity**: {severity}")
                    st.write(f"**Details**: {details}")
        else:
            st.info("No incidents recorded yet.")

    def _format_alert_message(self, detection_type, details=None):
        """Generate precise alert messages"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"{detection_type.capitalize()} Threat Alert\n"
        # Use default values if details is None or not a dict
        if not isinstance(details, dict):
            details = {"threat_type": "unknown", "confidence": 0, "reason": "No details available"}
        message += f"Type: {details.get('threat_type', 'unknown').upper()}\n"
        message += f"Confidence: {details.get('confidence', 0)}%\n"
        message += f"Analysis: {details.get('reason', 'No details')}\n"
        message += f"Timestamp: {timestamp}"
        return message

    def _send_alert(self, detection_type, filename, details=None):
        contacts = st.session_state.emergency_contact
        email = contacts.get('email')
        phone = contacts.get('phone')

        # Prepare alert details before formatting message
        if detection_type == "video":
            alert_details = {"threat_type": "violence", "confidence": 95, "reason": f"Violence detected in video: {filename}"}
        elif detection_type == "audio" and isinstance(details, dict):
            alert_details = details
        else:
            alert_details = {"threat_type": "unknown", "confidence": 0, "reason": "No details available"}

        # Format message with prepared details
        alert_msg = self._format_alert_message(detection_type, alert_details)

        # Send email alert if email is provided
        if email:
            success, message = self.alert_mail_system.send_alert(alert_msg, email)
            if success:
                st.success(f"Email alert sent to {email}")
            else:
                st.error(f"Failed to send email alert: {message}")
                self.logger.log_incident("ERROR: email_alert", "high", message)

        # Send SMS alert if phone is provided
        if phone:
            sms_msg = self.alert_sms_system.format_emergency_message(detection_type, alert_details)
            success, message = self.alert_sms_system.send_alert(sms_msg, phone)
            if success:
                st.success(f"SMS alert sent to {phone}")
            else:
                st.error(f"Failed to send SMS alert: {message}")
                self.logger.log_incident("ERROR: sms_alert", "high", message)

if __name__ == "__main__":
    app = ShieldVisionApp()
    app.run()