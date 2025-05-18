import os
from datetime import datetime
import json

class Logger:
    def __init__(self, log_file='shield_vision.log', settings_file='settings.txt'):
        self.log_file = log_file
        self.settings_file = settings_file
        # Initialize settings.txt with default JSON if it doesn't exist or is empty
        if not os.path.exists(self.settings_file) or os.path.getsize(self.settings_file) == 0:
            with open(self.settings_file, 'w') as f:
                json.dump({'email': '', 'phone': ''}, f)

    def log_incident(self, incident_type, severity, details):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(details, dict):
            details = json.dumps(details)
        log_entry = f"[{timestamp}] {incident_type} | Severity: {severity} | Details: {details}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        return timestamp

    def get_recent_incidents(self, limit=5):
        incidents = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[-limit:]
            for line in lines:
                try:
                    timestamp = line.split(']')[0][1:]
                    parts = line.split(' | ')
                    incident_type = parts[0].split('] ')[1]
                    severity = parts[1].split(': ')[1]
                    details = parts[2].split(': ')[1].strip()
                    incidents.append((timestamp, incident_type, severity, details))
                except:
                    continue
        return incidents

    def get_emergency_contact(self, contact_type='email'):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    content = f.read().strip()
                    if not content:  # Handle empty file
                        settings = {'email': '', 'phone': ''}
                        with open(self.settings_file, 'w') as fw:
                            json.dump(settings, fw)
                        return settings.get(contact_type, '')
                    settings = json.load(f)
                    return settings.get(contact_type, '')
            except json.JSONDecodeError:
                # If JSON is invalid, reset to default
                settings = {'email': '', 'phone': ''}
                with open(self.settings_file, 'w') as f:
                    json.dump(settings, f)
                return settings.get(contact_type, '')
        return ''

    def update_emergency_contact(self, contact, contact_type='email'):
        settings = {'email': self.get_emergency_contact('email'), 'phone': self.get_emergency_contact('phone')}
        settings[contact_type] = contact
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)