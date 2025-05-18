from dotenv import load_dotenv
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

class AlertSystem:
    def __init__(self, api_type="simulation", groq_api_key=None, smtp_server="smtp.gmail.com",
                 smtp_port=587, smtp_user=None, smtp_password=None, debug=False):
        # Initialization code
        self.api_type = api_type
        self.debug = debug
        self._initialize_credentials()
        self._validate_configuration()

    def _initialize_credentials(self):
        """Load and validate all required credentials"""
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.sender_name = os.getenv("SENDER_NAME", "SHIELD VISION Alert System")

    def _validate_configuration(self):
        """Validate the system configuration"""
        if self.api_type == "email":
            missing = []
            if not self.groq_api_key: missing.append("GROQ_API_KEY")
            if not self.smtp_user: missing.append("SMTP_USER")
            if not self.smtp_password: missing.append("SMTP_PASSWORD")

            if missing:
                print(f"Missing credentials: {', '.join(missing)}. Falling back to simulation.")
                self.api_type = "simulation"
            else:
                try:
                    self._initialize_email_components()
                    print("Email alert system initialized successfully")
                except Exception as e:
                    print(f"Failed to initialize email system: {str(e)}")
                    self.api_type = "simulation"

    def _initialize_email_components(self):
        """Initialize email-related components"""
        try:
            self.model = ChatGroq(model="llama-3.1-8b-instant", api_key=self.groq_api_key)
            self.parser = StrOutputParser()
            self.prompt = ChatPromptTemplate.from_template(
                """Write a professional email to notify about a SHIELD VISION alert for {detection_type} detected at {timestamp}.
                Include the following details: {details}.
                The email should be concise, urgent, and professional, urging immediate action."""
            )
            self.chain = self.prompt | self.model | self.parser
            self.groq_available = True
        except Exception as e:
            print(f"Failed to initialize Groq API: {e}")
            self.groq_available = False

    def send_alert(self, message, recipient_email):
        """Send an alert to the specified recipient"""
        if not self._validate_email(recipient_email):
            return False, "Invalid recipient email address"

        if self.api_type == "email":
            return self._send_email_alert(message, recipient_email)
        return self._simulate_alert(message, recipient_email)

    def _send_email_alert(self, message, recipient_email):
        """Send an email alert with proper error handling"""
        try:
            # Generate email content
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Try to use Groq API if available, otherwise use a template
            if hasattr(self, 'groq_available') and self.groq_available:
                try:
                    email_content = self.chain.invoke({
                        "detection_type": message.split(":")[0].lower(),
                        "timestamp": timestamp,
                        "details": message
                    })
                except Exception as e:
                    print(f"Groq API error when generating email: {e}")
                    # Fallback to template if Groq fails
                    email_content = self._generate_template_email(message, timestamp)
            else:
                # Use template if Groq is not available
                email_content = self._generate_template_email(message, timestamp)

            # Create email message
            msg = MIMEText(email_content)
            msg['Subject'] = f"SHIELD VISION ALERT - {timestamp}"
            msg['From'] = formataddr((self.sender_name, self.smtp_user))
            msg['To'] = recipient_email

            # Send email with secure connection
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return True, "Email sent successfully"

        except smtplib.SMTPAuthenticationError:
            return False, "SMTP Authentication failed. Check your email credentials."
        except smtplib.SMTPException as e:
            return False, f"SMTP Error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def _validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _simulate_alert(self, message, contact):
        """Simulate sending an alert"""
        time.sleep(1)
        print(f"SIMULATION: Alert sent to {contact}: {message}")
        return True, "Alert simulated"

    def _generate_template_email(self, message, timestamp):
        """Generate a template email when Groq API is not available"""
        # Extract detection type from the message
        detection_type = "alert"
        if ":" in message:
            detection_type = message.split(":")[0].lower()

        email_content = f"""URGENT: SHIELD VISION ALERT - {timestamp}

Dear Emergency Contact,

This is an automated alert from the SHIELD VISION safety system.

A potential threat has been detected:
{message}

Please take immediate action to verify this alert and ensure safety.

This is an automated message. Please do not reply to this email.

Regards,
SHIELD VISION Alert System
"""
        return email_content

    def format_emergency_message(self, detection_type, details=None):
        """Format an emergency alert message based on detection type"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"{detection_type.capitalize()} Threat Detected"

        if detection_type == "video":
            message += ": Violence detected in video footage."
        elif detection_type == "audio":
            message += ": Suspicious audio detected."
            if details and 'keywords' in details:
                message += f" Keywords: {', '.join(details['keywords'])}"

        message += f" Timestamp: {timestamp}. Immediate action required."
        return message


if __name__ == "__main__":
    # Initialize with debug output
    print("Initializing alert system...")
    alert_system = AlertSystem(api_type="email", debug=True)

    # Test with your email
    test_email = "shadan.anwar2005@gmail.com"
    message = "video: Suspicious activity detected in camera feed"

    print(f"\nSending test alert to {test_email}...")
    success, response = alert_system.send_alert(message, test_email)

    if success:
        print("SUCCESS:", response)
    else:
        print("FAILED:", response)
        print("\nTroubleshooting Steps:")
        print("1. Verify your SMTP credentials in .env file")
        print("2. Ensure you're using an App Password if 2FA is enabled")
        print("3. Check Google Account security settings:")
        print("   - https://myaccount.google.com/security")
        print("4. Try enabling 'Less secure app access'")
        print("5. Wait a few minutes after changing settings before retrying")