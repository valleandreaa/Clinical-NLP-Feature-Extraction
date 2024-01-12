from database import *

def insert_patient():
    first_name = input("Enter patient's first name: ")
    last_name = input("Enter patient's last name: ")
    age = input("Enter patient's age: ")
    gender = input("Enter patient's gender (Male/Female/Other): ")
    contact_information = input("Enter patient's contact information: ")
    emergency_contact = input("Enter patient's emergency contact: ")

    query = """
    INSERT INTO patients (first_name, last_name, age, gender, contact_information, emergency_contact)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (first_name, last_name, age, gender, contact_information, emergency_contact))
    conn.commit()
    print("Patient added successfully.")

insert_patient()
