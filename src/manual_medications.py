from database import *

def insert_medication():
    try:
        # Input medication details
        patient_id = input("Enter the patient ID: ")
        medication_name = input("Enter the medication name: ")
        dosage = input("Enter the medication dosage (e.g., '10mg', '2 tablets'): ")
        purpose = input("Enter the purpose of the medication: ")

        # Execute SQL query
        query = """
        INSERT INTO medications_table (patient_id, medication_name, dosage, purpose)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (patient_id, medication_name, dosage, purpose))
        conn.commit()
        print("Medication added successfully.")
    except Exception as e:
        print(f"Error inserting medication: {e}")

def main():
    # Establish connection
    conn = connect_to_database()
    if conn:
        cursor = conn.cursor()
        # Call insert_medication function
        insert_medication(conn, cursor)


if __name__ == "__main__":
    main()