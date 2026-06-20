import csv
import os
from datetime import datetime, date
import mysql.connector
from mysql.connector import Error
import numpy as np
import config
import faiss
import pickle

def get_connection():
    return mysql.connector.connect(
        host=config.DB_HOST,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME
    )

def insert_user_embedding(reg_no, name, embeddings):
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("DELETE FROM user_embeddings WHERE reg_no = %s", (reg_no,))

        for emb in embeddings:
            query = """INSERT INTO user_embeddings 
                       (reg_no, name, angle, embedding) 
                       VALUES (%s, %s, %s, %s)"""
            embedding_blob = pickle.dumps(emb['embedding'])
            cursor.execute(query, (reg_no, name, emb['angle'], embedding_blob))

        connection.commit()
        print(f"[INFO] User '{name}' registered with {len(embeddings)} angle embeddings")
        return True
    except Error as e:
        print(f"[ERROR] Inserting user embedding: {e}")
        return False
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'connection' in locals() and connection.is_connected(): connection.close()

def load_all_user_embeddings():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        query = "SELECT reg_no, name, angle, embedding FROM user_embeddings"
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"[ERROR] Loading embeddings: {e}")
        return []
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'connection' in locals() and connection.is_connected(): connection.close()

def mark_attendance(reg_no, name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    csv_filename = f'attendance_{date_str}.csv'

    print(f"[DEBUG] mark_attendance() called for {name} ({reg_no}) at {time_str}")

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check if already marked in DB
        query = """
            SELECT id FROM attendance_logs
            WHERE reg_no = %s AND DATE(timestamp) = CURDATE()
        """
        cursor.execute(query, (reg_no,))
        result = cursor.fetchone()
        if result:
            print(f"[INFO] Attendance already marked today for {name} ({reg_no})")
            return

        # Insert into DB
        insert_query = """
            INSERT INTO attendance_logs (reg_no, name, timestamp)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (reg_no, name, now))
        conn.commit()
        print(f"[INFO] Attendance marked in DB for {name} ({reg_no})")

        # Create CSV with all users marked Absent if not exists
        users = _get_all_registered_users()
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Reg No', 'Name', 'Date', 'Time', 'Status'])
                for user in users:
                    writer.writerow([user['reg_no'], user['name'], '', '', 'Absent'])

        # Update specific user to Present in CSV
        rows = []
        with open(csv_filename, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        for i in range(1, len(rows)):
            if rows[i][0] == reg_no:
                rows[i][2] = date_str
                rows[i][3] = time_str
                rows[i][4] = 'Present'
                break

        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"[INFO] Attendance updated in CSV for {name} ({reg_no})")

    except Exception as e:
        print(f"[ERROR] Failed to mark attendance for {name} ({reg_no}): {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals() and conn.is_connected(): conn.close()


def _update_csv_attendance(reg_no, name):
    today = date.today().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"

    users = _get_all_registered_users()

    if not os.path.exists(filename):
        # First time creation: mark all as Absent
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Reg No', 'Name', 'Status'])
            for user in users:
                status = 'Present' if user['reg_no'] == reg_no else 'Absent'
                writer.writerow([user['reg_no'], user['name'], status])
    else:
        # Update existing CSV: set this user to Present
        rows = []
        found = False
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        for i in range(1, len(rows)):
            if rows[i][0] == reg_no:
                rows[i][2] = 'Present'
                found = True

        if not found:
            rows.append([reg_no, name, 'Present'])

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

def _get_all_registered_users():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT reg_no, name FROM user_embeddings")
        results = cursor.fetchall()
        return [{'reg_no': r[0], 'name': r[1]} for r in results]
    except Error as e:
        print(f"[ERROR] Fetching users: {e}")
        return []
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'connection' in locals() and connection.is_connected(): connection.close()
