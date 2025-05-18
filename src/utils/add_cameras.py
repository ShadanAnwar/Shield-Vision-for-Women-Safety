import sqlite3

def update_cameras_table():
    try:
        # Connect to the database
        conn = sqlite3.connect('safe_women.db')
        cursor = conn.cursor()

        # Clear the existing entries in the cameras table
        cursor.execute("DELETE FROM cameras")
        print("Cleared existing entries in cameras table.")

        # Add the new camera feeds
        new_cameras = [
            ('Fresno Traffic Cam', 'Fresno, CA', 'https://www.webcamtaxi.com/en/usa/california/fresno-traffic-cam.html'),
            ('Duval Street Cam', 'Key West, FL', 'https://www.webcamtaxi.com/en/usa/florida/key-west-duval-street.html'),
            ('Rufilyn & Majah BBQ Cam', 'Davao, Philippines', 'https://www.webcamtaxi.com/en/philippines/davao-region/rufilyn-majah-bbq-cam.html'),
        ]

        cursor.executemany("INSERT INTO cameras (name, location, stream_url) VALUES (?, ?, ?)", new_cameras)
        conn.commit()
        print(f"Added {len(new_cameras)} new cameras to the database.")

        # Verify the updated table
        cursor.execute("SELECT * FROM cameras")
        updated_cameras = cursor.fetchall()
        print("Current entries in cameras table:")
        for camera in updated_cameras:
            print(camera)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# Run the function
update_cameras_table()