import os
from datetime import datetime
import json

class Logger:
    def __init__(self, log_file='shield_vision.log', settings_file='settings.txt'):
        self.log_file = log_file
        self.settings_file = settings_file
        if not os.path.exists(self.settings_file):
            with open(self.settings_file, 'w') as f:
                f.write('')

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
                lines = f.readlines()[-limit:]  # Read last 'limit' lines
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

    def get_emergency_contact(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                return f.read().strip()
        return ''

    def update_emergency_contact(self, phone):
        with open(self.settings_file, 'w') as f:
            f.write(phone)