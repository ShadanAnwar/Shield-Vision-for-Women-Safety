import os
import wave
import audioop
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class AudioAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Audio property thresholds - adjusted for better sensitivity
        self.SHOUTING_RMS_THRESHOLD = 2200  # Lowered threshold to detect more shouting
        self.NORMAL_SPEECH_MAX = 1800       # Maximum RMS for normal speech
        self.HIGH_PITCH_THRESHOLD = 280     # Slightly lowered to detect more high-pitch sounds
        self.LOW_PITCH_THRESHOLD = 85       # Maximum frequency for normal speech

        # Contextual analysis parameters
        self.MIN_CONFIDENCE = 75             # Minimum confidence to consider as threat

    def _analyze_audio_properties(self, audio_file):
        """Analyze raw audio properties with strict thresholds"""
        try:
            # Check file extension to determine how to process it
            file_ext = os.path.splitext(audio_file)[1].lower()

            if file_ext == '.mp3':
                # For MP3 files, we'll use a simulated analysis since we can't directly process them
                # In a real implementation, you would use a library like pydub to convert MP3 to WAV
                print(f"MP3 file detected: {audio_file}. Using simulated audio properties.")
                return {
                    'rms': 3500,  # Simulated high volume (above SHOUTING_RMS_THRESHOLD)
                    'dominant_freq': 320,  # Simulated high pitch (above HIGH_PITCH_THRESHOLD)
                    'is_shouting': True,  # Assume shouting for demonstration
                    'is_high_pitch': True,  # Assume high pitch for demonstration
                    'is_normal_pitch': False,
                    'is_normal_volume': False
                }

            # For WAV files, use the standard analysis
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                rms = audioop.rms(frames, wf.getsampwidth())

                # Frequency analysis using FFT
                raw_data = np.frombuffer(frames, dtype=np.int16)
                if len(raw_data) == 0:
                    return self._default_audio_props()

                fft = np.fft.fft(raw_data)
                frequencies = np.fft.fftfreq(len(fft), 1.0/wf.getframerate())
                dominant_freq = abs(frequencies[np.argmax(np.abs(fft[1:])) + 1])

                return {
                    'rms': rms,
                    'dominant_freq': dominant_freq,
                    'is_shouting': rms > self.SHOUTING_RMS_THRESHOLD,
                    'is_high_pitch': dominant_freq > self.HIGH_PITCH_THRESHOLD,
                    'is_normal_pitch': self.LOW_PITCH_THRESHOLD <= dominant_freq <= self.HIGH_PITCH_THRESHOLD,
                    'is_normal_volume': rms <= self.NORMAL_SPEECH_MAX
                }
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return self._default_audio_props()

    def _default_audio_props(self):
        """Return safe defaults when analysis fails"""
        return {
            'rms': 0,
            'dominant_freq': 0,
            'is_shouting': False,
            'is_high_pitch': False,
            'is_normal_pitch': True,
            'is_normal_volume': True
        }

    def _contextual_analysis(self, audio_props):
        """Use Groq to determine if shouting is actually threatening"""
        prompt = f"""
        Analyze these audio characteristics for potential threats:
        - Volume Level: {'Very High (shouting)' if audio_props['is_shouting'] else
                        'Normal' if audio_props['is_normal_volume'] else 'High'}
        - Pitch: {'High (screaming)' if audio_props['is_high_pitch'] else 'Normal'}
        - Dominant Frequency: {audio_props['dominant_freq']:.2f} Hz

        Consider that shouting could be from:
        - Excited non-threatening situations (sports, celebrations)
        - Actual distress or violence

        Return JSON response with:
        {{
            "is_threat": bool,
            "confidence": 0-100,
            "threat_type": "violence", "distress", or "none",
            "reason": "explanation of analysis"
        }}
        """

        # First try with Groq API
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Updated to a supported model
                response_format={"type": "json_object"},
                temperature=0.3  # More deterministic output
            )
            result = eval(response.choices[0].message.content)

            # Only consider as threat if confidence is high
            if result.get('confidence', 0) >= self.MIN_CONFIDENCE:
                return result
            return {
                "is_threat": False,
                "confidence": result.get('confidence', 0),
                "threat_type": "none",
                "reason": "Low confidence in threat detection"
            }
        except Exception as e:
            print(f"Groq analysis error: {e}")

            # Fallback to rule-based analysis if Groq API fails
            return self._rule_based_analysis(audio_props)

    def _rule_based_analysis(self, audio_props):
        """Fallback rule-based analysis when Groq API fails"""
        # Simple rule-based threat detection
        is_threat = False
        threat_type = "none"
        confidence = 0
        reason = "No threat detected based on audio characteristics."

        # If both shouting and high pitch, consider it a high-confidence threat
        if audio_props['is_shouting'] and audio_props['is_high_pitch']:
            is_threat = True
            threat_type = "distress"
            confidence = 90
            reason = "High volume shouting with high pitch detected, indicating potential distress."
        # If just shouting with very high volume
        elif audio_props['is_shouting'] and audio_props['rms'] > self.SHOUTING_RMS_THRESHOLD * 1.5:
            is_threat = True
            threat_type = "violence"
            confidence = 85
            reason = "Extremely high volume detected, indicating potential violence or emergency."
        # If just shouting (ANY shouting)
        elif audio_props['is_shouting']:
            is_threat = True
            threat_type = "distress"
            confidence = 80
            reason = "Shouting detected, indicating potential distress or emergency situation."
        # If just high pitch
        elif audio_props['is_high_pitch'] and audio_props['dominant_freq'] > self.HIGH_PITCH_THRESHOLD * 1.2:
            is_threat = True
            threat_type = "distress"
            confidence = 75
            reason = "Very high pitch detected, indicating potential distress call."

        return {
            "is_threat": is_threat,
            "confidence": confidence,
            "threat_type": threat_type,
            "reason": reason
        }

    def analyze_audio(self, audio_file):
        """Main analysis with strict threat verification"""
        audio_props = self._analyze_audio_properties(audio_file)

        # Try rule-based analysis first for reliability
        rule_analysis = self._rule_based_analysis(audio_props)

        # If rule-based analysis detected a threat, use it
        if rule_analysis["is_threat"]:
            analysis = rule_analysis
        else:
            # Otherwise try the contextual analysis (which may fall back to rule-based anyway)
            try:
                analysis = self._contextual_analysis(audio_props)
            except Exception as e:
                print(f"Error in contextual analysis: {e}")
                analysis = rule_analysis

        return {
            "is_threat": analysis["is_threat"],
            "threat_type": analysis["threat_type"],
            "confidence": analysis["confidence"],
            "reason": analysis["reason"],
            "audio_properties": audio_props
        }