import torch
from transformers import pipeline
from database  import *

nlp_ner = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner")

query = "SELECT * FROM clinical_notes"
cursor.execute(query)
rows = cursor.fetchall()

for pn_num, patient_id, case_num, clinical_text in rows:
    ner_results = nlp_ner(clinical_text)

    current_entity = None

    current_entities = []
    current_entity_words = []
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

        # If there was an ongoing entity, process and store it


    current_entities_clean=[]
    for entity_data in current_entities:
        entity_data['text'] = entity_data['text'].replace(' ##', '')
        current_entities_clean.append(entity_data)
    # Store the last entity (if any)

    # Insert entities into the clinical_analysis_table
    for entity_data in current_entities_clean:
        query_insert = "INSERT INTO clinical_analysis_table (note_id, entity, text, date_analysis) VALUES (%s, %s, %s, NOW())"
        cursor.execute(query_insert, (pn_num, str(entity_data['entity']), str(entity_data['text'])))
        conn.commit()