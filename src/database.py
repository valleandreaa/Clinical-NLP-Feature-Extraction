import mysql.connector

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
    host="86.119.40.9",
    user="admin",
    password="Xvn239vn$mACo92!",
    database="clinical_notes"
)

# Create a cursor object to execute SQL queries
cursor = conn.cursor()