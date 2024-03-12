import mysql.connector
from mysql.connector import Error

# Connection DB
def connect_to_database():
    try:
        conn = mysql.connector.connect(
            host="86.119.40.9",
            user="admin",
            password="Xvn239vn$mACo92!",
            database="clinical_notes"
        )
        if conn.is_connected():
            print("Connected to clinical_notes database")
            return conn

    except Error as e:
        print(f"Error connecting to clinical_notes database: {e}")

    return None

def close_connection(conn, cursor=None):
    try:
        if cursor:
            cursor.close()
        if conn.is_connected():
            conn.close()
            print("clinical_notes database connection closed")
    except Error as e:
        print(f"Error closing clinical_notes database connection: {e}")

conn = connect_to_database()

if conn:
    cursor = conn.cursor()