from database import *

def insert_medication():
    patient_id = input("Enter the patient ID: ")
    medication_name = input("Enter the medication name: ")
    dosage = input("Enter the medication dosage (e.g., '10mg', '2 tablets'): ")
    purpose = input("Enter the purpose of the medication: ")

    query = """
    INSERT INTO medications_table (patient_id, medication_name, dosage, purpose)
    VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (patient_id, medication_name, dosage, purpose))
    conn.commit()
    print("Medication added successfully.")

insert_medication()