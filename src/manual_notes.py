from transformers import pipeline
from database  import *


def insert_appointment():
    patient_id = input("Enter patient ID: ")
    date_time = input("Enter appointment date and time (YYYY-MM-DD HH:MM:SS): ")
    purpose = input("Enter the purpose of the appointment: ")
    notes = input("Enter any notes for the appointment: ")

    query = """
    INSERT INTO appointments_table (patient_id, date_time, purpose, notes)
    VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (patient_id, date_time, purpose, notes))
    conn.commit()
    print("Appointment added successfully.")

insert_appointment()



nlp_ner = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner")

# Fetch the latest note from appointments_table
query = "SELECT appointment_id, patient_id, notes FROM appointments_table ORDER BY appointment_id DESC LIMIT 1"
cursor.execute(query)
row = cursor.fetchone()

if row:
    appointment_id, patient_id, notes = row
    ner_results = nlp_ner(notes)

    current_entity = None
    current_entity_words = []
    current_entities = []

    for entity in ner_results:
        if entity['entity'].startswith('B-'):


            if current_entity:
                current_entities.append({'entity': current_entity, 'text': ' '.join(current_entity_words)})
            current_entity_words = []
            # Start a new entity
            current_entity = entity['entity'][2:]  # Remove the 'B-'
            current_entity_words = [entity['word']]


        elif entity['entity'].startswith('I-'):
            # If there's a continuing entity, add it to the current entity's words
            if current_entity and current_entity == entity['entity'][2:]:
                current_entity_words.append(entity['word'])

    current_entities.append({'entity': current_entity, 'text': ' '.join(current_entity_words)})
    current_entities_clean=[]
    for entity_data in current_entities:
        entity_data['text'] = entity_data['text'].replace(' ##', '')
        current_entities_clean.append(entity_data)
    # Store entities in the database
    for entity_data in current_entities_clean:
        # Modify this part to insert into the appropriate table
        query_insert = "INSERT INTO clinical_analysis_table (note_id, entity, text, date_analysis) VALUES (%s, %s, %s, NOW())"
        cursor.execute(query_insert, (patient_id, str(entity_data['entity']), str(entity_data['text'])))
        conn.commit()


cursor.close()
conn.close()