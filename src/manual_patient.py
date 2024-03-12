from database import *

def insert_patient(conn, cursor):
    try:
        # Input patient details
        first_name = input("Enter patient's first name: ")
        last_name = input("Enter patient's last name: ")
        age = int(input("Enter patient's age: "))  # Ensure age is an integer
        gender = input("Enter patient's gender (Male/Female/Other): ").capitalize()  # Capitalize the gender
        contact_information = input("Enter patient's contact information: ")
        emergency_contact = input("Enter patient's emergency contact: ")

        # Execute SQL query
        query = """
        INSERT INTO patients (first_name, last_name, age, gender, contact_information, emergency_contact)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (first_name, last_name, age, gender, contact_information, emergency_contact))
        conn.commit()
        print("Patient added successfully.")
    except ValueError:
        print("Error: Please enter a valid age.")
    except Exception as e:
        print(f"Error adding patient: {e}")

def main():
    # Establish connection
    conn = connect_to_database()
    if conn:
        cursor = conn.cursor()

        # Insert patient
        insert_patient(conn, cursor)

        # Close connection
        close_connection(conn, cursor)

if __name__ == "__main__":
    main()