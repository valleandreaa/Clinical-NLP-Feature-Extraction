from transformers import pipeline
from database  import *

def insert_medical_conditions(cursor, rows):
    try:
        # Insert data into medical_conditions_table
        insert_query = """
        INSERT INTO medical_conditions_table (patient_id, condition_name, status)
        VALUES (%s, %s, %s)
        """

        # Iterate over the results and insert each into the medical_conditions_table
        for patient_id, condition_text in rows:
            status = True
            cursor.execute(insert_query, (patient_id, condition_text, status))

        print("Medical conditions inserted successfully.")
    except Exception as e:
        print(f"Error inserting medical conditions: {e}")

def main():
    try:
        # Establish connection
        conn = connect_to_database()
        if conn:
            cursor = conn.cursor()

            # SQL query to find problems and their corresponding patient_id
            query = """
            SELECT cn.patient_id, cat.text
            FROM clinical_analysis_table cat
            JOIN clinical_notes cn ON cat.note_id = cn.pn_num
            WHERE cat.entity = 'problem'
            """

            # Execute the query
            cursor.execute(query)
            rows = cursor.fetchall()

            # Insert medical conditions
            insert_medical_conditions(cursor, rows)

            # Commit the changes to the database
            conn.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the cursor and connection
        close_connection(conn, cursor)

if __name__ == "__main__":
    main()