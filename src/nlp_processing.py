import torch
from transformers import pipeline
from database  import *

def extract_entities_and_store(conn, cursor):
    try:
        nlp_ner = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner")

        # Fetch clinical notes from the database
        query = "SELECT pn_num, clinical_text FROM clinical_notes"
        cursor.execute(query)
        rows = cursor.fetchall()

        for pn_num, clinical_text in rows:
            ner_results = nlp_ner(clinical_text)

            current_entities = []
            current_entity_words = []
            current_entity = None

            for entity in ner_results:
                if entity['entity'].startswith('B-'):
                    if current_entity:
                        current_entities.append({'entity': current_entity, 'text': ' '.join(current_entity_words)})
                    current_entity_words = []
                    current_entity = entity['entity'][2:]  # Remove the 'B-'
                    current_entity_words = [entity['word']]
                elif entity['entity'].startswith('I-'):
                    if current_entity and current_entity == entity['entity'][2:]:
                        current_entity_words.append(entity['word'])

            if current_entity:
                current_entities.append({'entity': current_entity, 'text': ' '.join(current_entity_words)})

            current_entities_clean = []
            for entity_data in current_entities:
                entity_data['text'] = entity_data['text'].replace(' ##', '')
                current_entities_clean.append(entity_data)

            # Insert entities into the clinical_analysis_table
            for entity_data in current_entities_clean:
                query_insert = "INSERT INTO clinical_analysis_table (note_id, entity, text, date_analysis) VALUES (%s, %s, %s, NOW())"
                cursor.execute(query_insert, (pn_num, str(entity_data['entity']), str(entity_data['text'])))
            conn.commit()

        print("Entities extraction and storage completed successfully.")
    except Exception as e:
        print(f"Error extracting entities and storing: {e}")

def main():
    # Establish connection
    conn = connect_to_database()
    if conn:
        cursor = conn.cursor()

        # Extract entities from clinical notes and store
        extract_entities_and_store(conn, cursor)

        # Close connection
        close_connection(conn, cursor)

if __name__ == "__main__":
    main()