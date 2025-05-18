from flask import Flask, jsonify, render_template, request, redirect, url_for, session, Response, flash
import sqlite3
import os
import torch
import cv2
from ultralytics import YOLO
from functools import wraps
import threading
import time
import hashlib
import re
import requests
from flask_mail import Mail, Message
import logging
from os import environ
import dotenv
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

# SQLite Configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'safe_women.db')

def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize database tables"""
    try:
        db = get_db()
        cursor = db.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            emergency_contact TEXT,
            emergency_contact_verified INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create camera_feeds table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS camera_feeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            feed_name TEXT NOT NULL,
            feed_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE,
            UNIQUE(username, feed_name)
        )
        ''')

        # Create cameras table (for public feeds)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT,
            stream_url TEXT NOT NULL
        )
        ''')

        # Check if admin user exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_count = cursor.fetchone()[0]

        # Create admin user if it doesn't exist
        if admin_count == 0:
            admin_password = hashlib.sha256('password'.encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ('admin', admin_password, 'admin')
            )
            print("✅ Default admin user created")

        # Insert the new camera feeds (only if table is empty)
        cursor.execute("SELECT COUNT(*) FROM cameras")
        camera_count = cursor.fetchone()[0]
        if camera_count == 0:
            sample_cameras = [
                ('Fresno Traffic Cam', 'Fresno, CA', 'https://www.webcamtaxi.com/en/usa/california/fresno-traffic-cam.html'),
                ('Duval Street Cam', 'Key West, FL', 'https://www.webcamtaxi.com/en/usa/florida/key-west-duval-street.html'),
                ('Rufilyn & Majah BBQ Cam', 'Davao, Philippines', 'https://www.webcamtaxi.com/en/philippines/davao-region/rufilyn-majah-bbq-cam.html'),
            ]
            cursor.executemany(
                "INSERT INTO cameras (name, location, stream_url) VALUES (?, ?, ?)",
                sample_cameras
            )
            print("✅ New cameras added")

        db.commit()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        db.close()

# Initialize database
init_db()

# Update all database operations to use SQLite
def get_connection():
    """Get database connection (compatibility function)"""
    return get_db()

def return_connection(conn):
    """Close database connection (compatibility function)"""
    conn.close()

# Dictionary to store the camera feeds and their detection status
camera_feeds = {}
detection_status = {}
alerts = []

# Load YOLO model
try:
    model_path = "models/yolo11_assault.pt"
    model = YOLO(model_path)
    print(f"🔍 YOLO model loaded from {model_path}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        
        try:
            connection = get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT role FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()
            cursor.close()
            return_connection(connection)
            
            if not user or user['role'] != 'admin':
                flash('Admin access required for this page', 'error')
                return redirect(url_for('home'))
                
            return f(*args, **kwargs)
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
            return redirect(url_for('home'))
            
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if user and user['password'] == hashlib.sha256(password.encode()).hexdigest():
                session['username'] = username
                session['role'] = user['role']
                return redirect(url_for('home'))
            else:
                error = 'Invalid credentials. Please try again.'
        except Exception as e:
            error = f'Database error: {str(e)}'
        finally:
            db.close()
    
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Form validation
        if len(username) < 4:
            error = 'Username must be at least 4 characters long.'
        elif not re.match(r'^[a-zA-Z0-9_]+$', username):
            error = 'Username can only contain letters, numbers, and underscores.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters long.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            try:
                connection = get_connection()
                cursor = connection.cursor()
                
                # Check if username exists
                cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
                user_exists = cursor.fetchone()
                
                if user_exists:
                    error = 'Username already exists. Please choose another one.'
                else:
                    # Add new user
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    cursor.execute(
                        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                        (username, hashed_password, 'user')
                    )
                    connection.commit()
                    
                    # Set success message
                    flash('Account created successfully! You can now log in.', 'success')
                    
                    cursor.close()
                    return_connection(connection)
                    return redirect(url_for('login'))
                
                cursor.close()
                return_connection(connection)
            except Exception as e:
                error = f'Database error: {str(e)}'
    
    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('index'))

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        feed_name = request.form['feed_name']
        feed_url = request.form['feed_url']
        
        try:
            connection = get_connection()
            cursor = connection.cursor()
            
            # Store the camera feed in the database
            cursor.execute(
                "INSERT INTO camera_feeds (username, feed_name, feed_url) VALUES (?, ?, ?)",
                (session['username'], feed_name, feed_url)
            )
            connection.commit()
            cursor.close()
            return_connection(connection)
            
            flash(f'Camera feed "{feed_name}" added successfully', 'success')
        except sqlite3.IntegrityError as err:
            flash(f'A camera with name "{feed_name}" already exists', 'error')
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
        
        return redirect(url_for('home'))
    
    # Get user's camera feeds from database
    camera_feeds_for_user = {}
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT feed_name, feed_url FROM camera_feeds WHERE username = ?",
            (session['username'],)
        )
        results = cursor.fetchall()
        cursor.close()
        return_connection(connection)
        
        for row in results:
            camera_feeds_for_user[row['feed_name']] = row['feed_url']
    except Exception as e:
        flash(f'Error loading camera feeds: {str(e)}', 'error')
    
    return render_template('home.html', camera_feeds=camera_feeds_for_user, role=session.get('role', 'user'))

from flask_cors import CORS

# Enable CORS for frontend access
CORS(app)


import requests

@app.route('/public_cameras')
def public_cameras():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT id, name, location, stream_url FROM cameras")
        cameras_raw = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        connection.close()

        # Validate each camera stream URL
        cameras = []
        for camera in cameras_raw:
            try:
                response = requests.head(camera['stream_url'], timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    camera['is_accessible'] = True
                else:
                    camera['is_accessible'] = False
            except requests.RequestException:
                camera['is_accessible'] = False
            cameras.append(camera)

        return render_template('public_cameras.html', cameras=cameras)
    except Exception as e:
        return render_template('public_cameras.html', cameras=[], error=str(e))
    

# Optional API endpoint for future use
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT id, name, location, stream_url FROM cameras")
        cameras = [
            {
                'id': row['id'],
                'name': row['name'],
                'location': row['location'],
                'stream_url': row['stream_url']
            } for row in cursor.fetchall()
        ]
        cursor.close()
        return_connection(connection)
        print(f"Returning cameras: {cameras}")  # Add logging
        return jsonify(cameras)
    except Exception as e:
        print(f"Error in get_cameras: {str(e)}")  # Add logging
        return jsonify({'error': f'Database error: {str(e)}'}), 500


@app.route('/users')
@admin_required
def manage_users():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT username, role, created_at FROM users")
        users = cursor.fetchall()
        cursor.close()
        return_connection(connection)
        return render_template('users.html', users=users)
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/users/delete/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    if username == 'admin':
        flash('Cannot delete the admin account', 'error')
        return redirect(url_for('manage_users'))
        
    if username == session['username']:
        flash('Cannot delete your own account', 'error')
        return redirect(url_for('manage_users'))
    
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        connection.commit()
        cursor.close()
        return_connection(connection)
        
        flash(f'User {username} deleted successfully', 'success')
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
    
    return redirect(url_for('manage_users'))

@app.route('/users/change_role/<username>', methods=['POST'])
@admin_required
def change_role(username):
    if username == 'admin':
        flash('Cannot change the role of the admin account', 'error')
        return redirect(url_for('manage_users'))
        
    role = request.form.get('role')
    if role not in ['admin', 'user']:
        flash('Invalid role', 'error')
        return redirect(url_for('manage_users'))
    
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        connection.commit()
        cursor.close()
        return_connection(connection)
        
        flash(f'Role for {username} updated to {role}', 'success')
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
    
    return redirect(url_for('manage_users'))

@app.route('/delete_camera/<feed_name>', methods=['POST'])
@login_required
def delete_camera(feed_name):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        # Delete the camera feed from the database
        cursor.execute(
            "DELETE FROM camera_feeds WHERE username = ? AND feed_name = ?",
            (session['username'], feed_name)
        )
        connection.commit()
        cursor.close()
        return_connection(connection)
        
        # Also remove from in-memory cache if present
        if feed_name in camera_feeds:
            del camera_feeds[feed_name]
        if feed_name in detection_status:
            del detection_status[feed_name]
        
        flash(f'Camera feed "{feed_name}" deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting camera feed: {str(e)}', 'error')
    
    return redirect(url_for('live_feed'))

@app.route('/emergency_contact', methods=['GET', 'POST'])
@login_required
def manage_emergency_contact():
    error = None
    success = None
    current_contact = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        if request.method == 'POST':
            phone_number = request.form['phone_number']
            
            # Validate phone number
            if not re.match(r'^\+?1?\d{10,14}$', phone_number):
                error = 'Invalid phone number format. Please include country code.'
            else:
                # Update emergency contact in database
                cursor.execute(
                    """UPDATE users 
                    SET emergency_contact = ?, 
                        emergency_contact_verified = 0 
                    WHERE username = ?""", 
                    (phone_number, session['username'])
                )
                connection.commit()
                
                # Send verification code (simulated here, you'd implement actual verification)
                verification_code = generate_verification_code()
                send_verification_sms(phone_number, verification_code)
                
                success = 'Emergency contact updated. Please verify your number.'
        
        # Retrieve current emergency contact
        cursor.execute(
            """SELECT emergency_contact, emergency_contact_verified 
            FROM users WHERE username = ?""", 
            (session['username'],)
        )
        user_info = cursor.fetchone()
        
        current_contact = {
            'number': user_info['emergency_contact'] if user_info else None,
            'verified': user_info['emergency_contact_verified'] if user_info else False
        }
        
        cursor.close()
        return_connection(connection)
    
    except Exception as e:
        error = f'Database error: {str(e)}'
    
    return render_template('emergency_contact.html', 
                           error=error, 
                           success=success, 
                           current_contact=current_contact)

@app.route('/verify_emergency_contact', methods=['GET', 'POST'])
@login_required
def verify_emergency_contact():
    error = None
    success = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Retrieve current user's emergency contact
        cursor.execute(
            """SELECT emergency_contact 
            FROM users WHERE username = ?""", 
            (session['username'],)
        )
        user_info = cursor.fetchone()

        if request.method == 'POST':
            verification_code = request.form['verification_code']
            
            # Verify the code (in a real implementation, you'd check against a stored/sent code)
            if verify_sms_code(verification_code):
                # Mark contact as verified
                cursor.execute(
                    """UPDATE users 
                    SET emergency_contact_verified = 1 
                    WHERE username = ?""", 
                    (session['username'],)
                )
                connection.commit()
                
                success = 'Emergency contact number verified successfully!'
            else:
                error = 'Invalid verification code. Please try again.'
        
        cursor.close()
        return_connection(connection)
    
    except Exception as e:
        error = f'Database error: {str(e)}'
    
    return render_template('verify_emergency_contact.html', 
                           error=error, 
                           success=success, 
                           contact_number=user_info['emergency_contact'] if user_info else None)

def generate_verification_code():
    """Generate a 6-digit verification code"""
    import random
    return str(random.randint(100000, 999999))

def send_verification_sms(phone_number, verification_code):
    """Send SMS with verification code"""
    message = f"Your verification code is: {verification_code}"
    return send_sms_alert(phone_number, message)

def verify_sms_code(code):
    """
    Verify SMS code 
    Note: In a real implementation, you'd store and check against the actual sent code
    """
    # Simulated verification - replace with actual verification logic
    return len(code) == 6 and code.isdigit()

# Modify send_sms_alert function to handle verification
def send_sms_alert(phone_number, message, server_url="https://textbelt.com/text"):
    """
    Send SMS alert using Textbelt API with improved error handling
    
    Args:
        phone_number (str): Phone number to send SMS to
        message (str): Message content
        server_url (str, optional): Textbelt server URL
    
    Returns:
        bool: True if SMS sent successfully, False otherwise
    """
    try:
        # Prepare payload for smschef API
        payload = {
            'key':'textbelt',
            'phone': phone_number,
            'message': message,
            # Optional parameters can be added here
        }
        
        # Send SMS via smschef API
        response = requests.post(url=server_url, data=payload, timeout=10)
        
        # Check response from smschef API
        response_data = response.json()
        
        if response.status_code == 200 and response_data.get('success', False):
            # Log successful SMS
            flash(f"SMS sent to {phone_number}")
            logging.info(f"SMS sent to {phone_number}")
            return True
        else:
            # Log failed SMS attempt
            flash(f"Failed to send SMS to {phone_number}")
            logging.warning(f"Failed to send SMS to {phone_number}. Response: {response_data}")
            return False
    
    except Exception as e:
        # Log any errors
        logging.error(f"Error sending SMS: {e}")
        return False

def get_location():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('loc', 'Unknown location')
        return location
    except Exception as e:
        print(f"Error getting location: {e}")
        return 'Unknown location'
    
def generate_frames(feed_name, emergency_contact):
    feed_url = camera_feeds[feed_name]
    cap = cv2.VideoCapture(feed_url)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {feed_url}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        return
    
    print(f"🔍 Starting detection for feed: {feed_name}")
    
    message_sent = False  # Move the flag outside the loop
    
    while True:
        success, frame = cap.read()
        if not success:
            # Try to reopen the connection in case of network issues
            cap = cv2.VideoCapture(feed_url)
            if not cap.isOpened():
                break
            continue
            
        # Apply YOLO assault detection if model is loaded
        if model is not None:
            results = model(frame)
            assault_detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    label = result.names[int(box.cls[0])]  # Get class name
                    confidence = float(box.conf[0])  # Get confidence score
                    
                    if (label != "People" and label != "Police") and confidence > 0.5:  
                        assault_detected = True
                        # Draw bounding box around detected assault
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Update detection status and add alert if assault is detected
            if assault_detected and not detection_status.get(feed_name, False):
                detection_status[feed_name] = True
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                location = get_location()
                alerts.append({
                    "feed_name": feed_name,
                    "timestamp": timestamp,
                    "location": location,
                    "message": f"🚨 ALERT! Assault detected in {feed_name}! Alert Sent to Emergency Contact"
                })
                
                if not message_sent:
                    send_sms_alert(emergency_contact, 
                         f"🚨 ALERT! Assault detected in {feed_name} at {timestamp} on location {location}")
                    message_sent = True

                
            elif not assault_detected:
                detection_status[feed_name] = False
                message_sent = False # Reset the flag if no assault is detected
                
            
            # Add alert indicator to frame if assault detected
            if detection_status.get(feed_name, False):
                cv2.putText(frame, "ASSAULT DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
        # Convert the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    print(f"🔴 Detection stopped for feed: {feed_name}")

@app.route('/video_feed/<feed_name>')
@login_required
def video_feed(feed_name):
    # Check if feed exists and belongs to user
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT feed_url, emergency_contact FROM camera_feeds JOIN users ON camera_feeds.username = users.username WHERE camera_feeds.username = ? AND feed_name = ?",
            (session['username'], feed_name)
        )
        feed = cursor.fetchone()
        cursor.close()
        return_connection(connection)
        
        if feed:
            # Get feed URL from database and store in memory cache for the detection process
            camera_feeds[feed_name] = feed['feed_url']
            
            return Response(generate_frames(feed_name,feed['emergency_contact']),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
    
    return "Feed not found", 404

@app.route('/alerts')
@login_required
def get_alerts():
    return {"alerts": alerts[-10:]}  # Return last 10 alerts

@app.route('/live_feed')
@login_required
def live_feed():
    # Get user's camera feeds from database
    camera_feeds_for_user = {}
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT feed_name, feed_url FROM camera_feeds WHERE username = ?",
            (session['username'],)
        )
        results = cursor.fetchall()
        cursor.close()
        return_connection(connection)
        
        for row in results:
            camera_feeds_for_user[row['feed_name']] = row['feed_url']
            # Also cache in memory for the detection process
            camera_feeds[row['feed_name']] = row['feed_url']
    except Exception as e:
        flash(f'Error loading camera feeds: {str(e)}', 'error')
    
    return render_template('live_feed.html', camera_feeds=camera_feeds_for_user, role=session.get('role', 'user'))

@app.route('/clear_alerts')
@login_required
def clear_alerts():
    global alerts
    alerts = []
    return redirect(url_for('live_feed'))

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    error = None
    success = None
    
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        try:
            connection = get_connection()
            cursor = connection.cursor()
            
            # Get current user
            cursor.execute("SELECT password FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()
            
            # Check if current password is correct
            if not user or user['password'] != hashlib.sha256(current_password.encode()).hexdigest():
                error = 'Current password is incorrect.'
            elif len(new_password) < 6:
                error = 'New password must be at least 6 characters long.'
            elif new_password != confirm_password:
                error = 'New passwords do not match.'
            else:
                # Update password
                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                cursor.execute(
                    "UPDATE users SET password = ? WHERE username = ?", 
                    (hashed_password, session['username'])
                )
                connection.commit()
                success = 'Password changed successfully!'
            
            cursor.close()
            return_connection(connection)
        except Exception as e:
            error = f'Database error: {str(e)}'
    
    return render_template('change_password.html', error=error, success=success)

dotenv.load_dotenv()
# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] =  os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] =  os.environ.get('MAIL_PASSWORD') # Use app password for Gmail
mail = Mail(app)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        try:
            msg = Message(
                subject=f'Contact Form: {subject}',
                sender=email,
                recipients=['shadan.anwar2005@gmail.com'],
                body=f'''
From: {name} <{email}>

{message}
'''
            )
            mail.send(msg)
            flash('Thank you for your message. We will get back to you soon!', 'success')
        except Exception as e:
            print(f"Error sending email: {e}")
            flash('Sorry, there was an error sending your message. Please try again.', 'error')
            
        return redirect(url_for('contact'))
        
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy_policy.html')
@app.route('/terms')
def terms():
    return render_template('terms_of_service.html')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, threaded=True)