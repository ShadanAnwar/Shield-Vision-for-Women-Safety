import sqlite3

def update_stream_urls():
    try:
        conn = sqlite3.connect('safe_women.db')
        cursor = conn.cursor()

        # Update with direct HLS stream URLs (replace these with the actual URLs you find)
        new_urls = [
            ('Fresno Traffic Cam', 'https://direct-stream-url-for-fresno.m3u8'),
            ('Duval Street Cam', 'https://direct-stream-url-for-duval.m3u8'),
            ('Rufilyn & Majah BBQ Cam', 'https://direct-stream-url-for-davao.m3u8'),
        ]

        for name, stream_url in new_urls:
            cursor.execute("UPDATE cameras SET stream_url = ? WHERE name = ?", (stream_url, name))

        conn.commit()
        print("Updated stream URLs in the database.")

        # Verify the updates
        cursor.execute("SELECT * FROM cameras")
        updated_cameras = cursor.fetchall()
        print("Current entries in cameras table:")
        for camera in updated_cameras:
            print(camera)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

update_stream_urls()