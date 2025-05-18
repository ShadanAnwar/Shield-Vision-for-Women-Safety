import os
from os.path import dirname, join
from dotenv import load_dotenv
import time
import re
from vonage import Auth, Vonage
from vonage_sms import SmsMessage, SmsResponse
import vonage  # For fallback to older API

class AlertSmsSystem:
    def __init__(self, api_type="sms", debug=True):
        self.api_type = api_type
        self.debug = debug
        self._initialize_credentials()
        self._validate_configuration()

    def _initialize_credentials(self):
        """Load Vonage credentials from environment variables"""
        dotenv_path = join(dirname(__file__), "../.env")
        load_dotenv(dotenv_path)
        self.vonage_key = os.getenv("VONAGE_API_KEY")
        self.vonage_secret = os.getenv("VONAGE_API_SECRET")
        self.sms_sender_id = os.getenv("SMS_SENDER_ID", "Vonage APIs")  # Default to match working snippet
        if self.debug:
            print(f"Using Vonage key: {self.vonage_key[:4]}**** if set, sender ID: {self.sms_sender_id}")

    def _validate_configuration(self):
        """Validate the system configuration"""
        if self.api_type == "sms":
            missing = []
            if not self.vonage_key:
                missing.append("VONAGE_API_KEY")
            if not self.vonage_secret:
                missing.append("VONAGE_API_SECRET")
            if missing:
                print(f"Missing credentials: {', '.join(missing)}. Falling back to simulation.")
                self.api_type = "simulation"
            else:
                try:
                    # Initialize new API
                    self.client = Vonage(Auth(api_key=self.vonage_key, api_secret=self.vonage_secret))
                    # Initialize fallback client for older API
                    self.fallback_client = vonage.Client(key=self.vonage_key, secret=self.vonage_secret)
                    self.fallback_sms = vonage.Sms(self.fallback_client)
                    print("Vonage SMS alert system initialized successfully")
                except Exception as e:
                    print(f"Failed to initialize Vonage SMS system: {str(e)}")
                    self.api_type = "simulation"
        else:
            self.api_type = "simulation"
            print("Falling back to simulation mode")

    def send_alert(self, message, phone_number):
        """Send an SMS alert to the specified phone number"""
        if not self._validate_phone(phone_number):
            return False, "Invalid phone number"

        if self.api_type == "sms":
            return self._send_sms_alert(message, phone_number)
        return self._simulate_alert(message, phone_number)

    def _send_sms_alert(self, message, phone_number):
        """Send an SMS alert using Vonage API"""
        try:
            # Match working snippet: remove +91 to use number as 919403308822
            if phone_number.startswith('+91'):
                phone_number = phone_number[3:]

            if self.debug:
                print(f"Sending SMS to: {phone_number}")
                print(f"Message: {message}")

            # Try new API first
            try:
                sms_message = SmsMessage(
                    to=phone_number,
                    from_=self.sms_sender_id,
                    text=message[:160]  # SMS standard limit
                )
                response: SmsResponse = self.client.sms.send(sms_message)

                # Access response attributes (not dictionary)
                if hasattr(response, 'messages') and response.messages:
                    status = response.messages[0].status
                    error_text = getattr(response.messages[0], 'error_text', 'Unknown error')
                    if status == "0":
                        if self.debug:
                            print("SMS sent successfully (new API)")
                        return True, "SMS sent successfully"
                    else:
                        if self.debug:
                            print(f"SMS failed (new API): {error_text}")
                        return False, f"SMS failed: {error_text}"
                else:
                    raise ValueError("Invalid response structure")

            except Exception as e:
                if self.debug:
                    print(f"New API failed: {str(e)}. Falling back to older API.")

                # Fallback to older API (working snippet)
                payload = {
                    "from": self.sms_sender_id,
                    "to": phone_number,
                    "text": message[:160]
                }
                response = self.fallback_sms.send_message(payload)
                if response["messages"][0]["status"] == "0":
                    if self.debug:
                        print("SMS sent successfully (fallback API)")
                    return True, "SMS sent successfully"
                else:
                    error = response["messages"][0].get("error-text", "Unknown error")
                    if self.debug:
                        print(f"SMS failed (fallback API): {error}")
                    return False, f"SMS failed: {error}"

        except Exception as e:
            if self.debug:
                print(f"Unexpected error: {str(e)}")
            return False, f"Unexpected error: {str(e)}"

    def _validate_phone(self, phone):
        """Validate phone number format (10 digits, with or without +91)"""
        cleaned_phone = phone.replace('+91', '') if phone.startswith('+91') else phone
        pattern = r'^\d{10}$'
        is_valid = re.match(pattern, cleaned_phone) is not None
        if self.debug and not is_valid:
            print(f"Invalid phone number: {phone}")
        return is_valid

    def _simulate_alert(self, message, phone):
        """Simulate sending an SMS alert"""
        time.sleep(1)
        print(f"SIMULATION: SMS sent to {phone}: {message}")
        return True, "SMS simulated"

    def format_emergency_message(self, detection_type, details=None):
        """Format an emergency SMS message based on detection type"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"SHIELD VISION ALERT: {detection_type.capitalize()} Threat Detected"

        if detection_type == "video":
            message += ": Violence in video."
        elif detection_type == "audio":
            message += ": Suspicious audio."
            if details and 'keywords' in details:
                message += f" Keywords: {', '.join(details['keywords'])}"

        message += f" Time: {timestamp}. Act now!"
        return message[:160]  # Ensure message fits within SMS limit

if __name__ == "__main__":
    # Initialize with debug output
    print("Initializing SMS alert system...")
    sms_system = AlertSmsSystem(api_type="sms", debug=True)

    # Test with phone number from previous context
    test_phone = "919403308822"  # Match working format
    message = "video: Suspicious activity detected"

    print(f"\nSending test SMS to {test_phone}...")
    success, response = sms_system.send_alert(message, test_phone)

    if success:
        print("SUCCESS:", response)
    else:
        print("FAILED:", response)
        print("\nTroubleshooting Steps:")
        print("1. Verify VONAGE_API_KEY and VONAGE_API_SECRET in .env match your Vonage Dashboard")
        print("2. Ensure the recipient number is verified in Vonage Dashboard (trial accounts)")
        print("3. Check Vonage account balance and trial restrictions")
        print("4. Confirm SMS_SENDER_ID is valid (e.g., 'Vonage APIs' for trial accounts)")
        print("5. Visit https://dashboard.nexmo.com for account status and message logs")