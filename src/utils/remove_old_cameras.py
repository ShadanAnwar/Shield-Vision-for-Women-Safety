import sqlite3

def keep_last_two_cameras():
    try:
        # Connect to the database
        conn = sqlite3.connect('safe_women.db')
        cursor = conn.cursor()

        # Get the total number of entries
        cursor.execute("SELECT COUNT(*) FROM cameras")
        total_entries = cursor.fetchone()[0]
        print(f"Total entries in cameras table: {total_entries}")

        if total_entries <= 2:
            print("There are 2 or fewer entries. No deletion needed.")
            return

        # Find the IDs of the last two entries
        cursor.execute("SELECT id FROM cameras ORDER BY id DESC LIMIT 2")
        last_two_ids = [row[0] for row in cursor.fetchall()]
        print(f"Keeping entries with IDs: {last_two_ids}")

        # Delete all entries except the last two
        cursor.execute("DELETE FROM cameras WHERE id NOT IN (?, ?)", (last_two_ids[0], last_two_ids[1]))

        # Commit the changes
        conn.commit()
        print(f"Deleted {total_entries - 2} entries. Kept the last 2 entries.")

        # Verify the remaining entries
        cursor.execute("SELECT * FROM cameras")
        remaining = cursor.fetchall()
        print("Remaining entries:")
        for row in remaining:
            print(row)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# Run the function
keep_last_two_cameras()