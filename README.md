# Shield Vision 🛡️

A women's safety system with two implementations:
1. **CCTV Monitoring System** - A Flask web application that monitors CCTV feeds for assault detection
2. **Media Analysis Tool** - A Streamlit application that analyzes video and audio files for threats

## Features

### CCTV Monitoring System
- **Real-time CCTV monitoring** using YOLO object detection
- **Assault detection** with automatic alerts
- **User management** with different role levels
- **Emergency contact system** with SMS alerts
- **Public camera feeds** integration

### Media Analysis Tool
- **Video threat detection** using MoBiLSTM model
- **Audio analysis** for detecting shouting, high pitch, and threatening sounds
- **Alert system** via email and SMS
- **Incident logging** for all detections

## Screenshots

### CCTV Monitoring System
<!-- Add your screenshots here -->

### Media Analysis Tool
<!-- Add your screenshots here -->

## Installation

### Prerequisites
- Python 3.8+
- YOLO model for assault detection
- MoBiLSTM model for video analysis
- SQLite3
- OpenCV

### Setup CCTV Monitoring System

1. Clone the repository
```bash
git clone https://github.com/your-username/shield-vision.git
cd shield-vision
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
Create a `.env` file in the root directory with:
```
SECRET_KEY=your_secret_key
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
```

5. Run the Flask application
```bash
python src/app.py
```
   
   ![Running Flask App](path_to_flask_screenshot.png)
   
   *Terminal showing Flask application running with the command: `python -u src\app.py`*

6. Access the application at `http://127.0.0.1:5000`

### Setup Media Analysis Tool

1. Navigate to the testing UI directory
```bash
cd "testing UI"
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements_streamlit.txt
```

4. Configure environment variables
Create a `.env` file in the testing UI directory with:
```
GROQ_API_KEY=your_groq_api_key
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

5. Run the Streamlit application
```bash
streamlit run app.py
```

   ![Running Streamlit App](path_to_streamlit_screenshot.png)
   
   *Terminal showing Streamlit application running with the command: `streamlit run .\app.py`*

6. Access the application at `http://localhost:8501`

## How It Works

### CCTV Monitoring System
1. The system connects to CCTV camera feeds
2. YOLO model processes each frame to detect assault
3. When assault is detected:
   - Visual alert appears on the interface
   - SMS alert is sent to emergency contact
   - Incident is logged in the system

### Media Analysis Tool
1. Upload video or audio files
2. The system analyzes content for threats:
   - Videos are processed by MoBiLSTM model
   - Audio is analyzed for shouting, high pitch, and threatening sounds
3. If threats are detected:
   - Results are displayed on the interface
   - Alerts are sent via email and SMS
   - Incident is logged in the system

## Dataset

The video analysis model was trained using the ["Real Life Violence Situations" dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) from Kaggle.

### Dataset Description

- **Content**: 2000 videos (1000 violence + 1000 non-violence)
- **Source**: YouTube videos 
- **Violence footage**: Real street fights in various environments and conditions
- **Non-violence footage**: Various human actions like sports, eating, walking, etc.

### Citation

If you use this dataset for research or engineering purposes, please cite:
```
M. Soliman, M. Kamal, M. Nashed, Y. Mostafa, B. Chawky, D. Khattab, 
"Violence Recognition from Videos using Deep Learning Techniques", 
Proc. 9th International Conference on Intelligent Computing and Information Systems (ICICIS'19), 
Cairo, pp. 79-84, 2019.
```

### Acknowledgements

Thanks to the dataset creators: 
- Dr. Dina Khattab (Supervisor)
- TA. Bassel Safwat (Supervisor)
- Team members: Mohamed Hussein, Mina Abd El-Massih, Youssif Mohamed

## Requirements

### Flask Application
```
flask==2.0.1
flask-mail==0.9.1
flask-cors==3.0.10
opencv-python==4.5.3.56
torch==1.9.0
ultralytics==8.0.0
requests==2.26.0
python-dotenv==0.19.0
```

### Streamlit Application
```
streamlit==1.12.0
numpy==1.21.2
pandas==1.3.3
tensorflow==2.6.0
keras==2.6.0
librosa==0.8.1
scikit-learn==0.24.2
python-dotenv==0.19.0
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Project Structure

```
shield-vision/
├── src/                          # Flask application source code
│   ├── app.py                    # Main Flask application
│   ├── templates/                # HTML templates
│   │   ├── index.html            # Landing page
│   │   ├── login.html            # Login page
│   │   ├── signup.html           # Registration page
│   │   ├── home.html             # User dashboard
│   │   ├── live_feed.html        # Live CCTV monitoring interface
│   │   ├── public_cameras.html   # Public camera feed page
│   │   ├── emergency_contact.html# Emergency contact management
│   │   ├── users.html            # User management (admin)
│   │   └── ...
│   ├── static/                   # Static files (CSS, JS, images)
│   └── models/                   # YOLO model for assault detection
│       └── yolo11_assault.pt     # Trained YOLO model
├── testing UI/                   # Streamlit application directory
│   ├── app.py                    # Main Streamlit application
│   ├── helpers/                  # Helper modules
│   │   ├── video_model.py        # Video analysis module
│   │   ├── audio_analyzer.py     # Audio analysis module
│   │   ├── alert_mail_system.py  # Email alert system
│   │   ├── alert_sms_system.py   # SMS alert system
│   │   └── logger.py             # Incident logging system
│   └── MoBiLSTM_best_model.keras # Trained MoBiLSTM model
├── requirements.txt              # Flask application dependencies
├── requirements_streamlit.txt    # Streamlit application dependencies
└── README.md                     # Project documentation
```

## Development Setup

### VS Code Setup
1. Open the project in VS Code
2. Install recommended extensions:
   - Python
   - Flask-Snippets
   - SQLite

### Running CCTV Monitoring System
1. Open a terminal in VS Code
2. Navigate to the project root directory
3. Run the application:
```bash
python -u src\app.py
```

### Running Media Analysis Tool
1. Open a terminal in VS Code
2. Navigate to the testing UI directory:
```bash
cd .\testing UI\
```
3. Activate the virtual environment:
```bash
.\.venv\Scripts\activate
```
4. Run the Streamlit application:
```bash
streamlit run .\app.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- [YOLO](https://github.com/ultralytics/yolov5) for object detection
- [Streamlit](https://streamlit.io/) for the user interface
- [Flask](https://flask.palletsprojects.com/) for the web application
