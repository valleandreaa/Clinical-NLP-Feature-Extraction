import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from database import *

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")

# Fetch data from clinical_notes
query = "SELECT * FROM clinical_notes"
cursor.execute(query)
rows = cursor.fetchall()

# Function to apply summarizer model to individual texts
def apply_summarizer(text):
    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="pt")
    summaries = model.generate(input_ids, max_length=230, min_length=30, do_sample=False)
    return tokenizer.decode(summaries[0], skip_special_tokens=True)

# Process each row
for pn_num, patient_id, case_num, clinical_text in rows:
    summary = apply_summarizer(clinical_text)

    # Insert the summary without character counts
    query_insert = "INSERT INTO notes_summarization_table (note_id, summary) VALUES (%s, %s)"
    cursor.execute(query_insert, (pn_num, summary))
    conn.commit()

    # Update the record with character counts
    query_update = """
    UPDATE notes_summarization_table
    SET chr_text = LENGTH(%s),
        chr_sum = LENGTH(%s)
    WHERE note_id = %s
    """
    cursor.execute(query_update, (clinical_text, summary, pn_num))
    conn.commit()








# def count_characters(text):
#     characters = text.split()
#     return len(characters)
#
# # Create new lists to store character counts
# note_counts = []
# summary_counts = []
#
# # Count the number of characters in each text and append to the lists
# for note_text, summary_text in zip(clinical_notes['notes'], clinical_notes['summary']):
#     note_count = count_characters(note_text)
#     summary_count = count_characters(summary_text)
#
#     note_counts.append(note_count)
#     summary_counts.append(summary_count)
#
# # Add the lists as new columns to the DataFrame
# clinical_notes['note_count'] = note_counts
# clinical_notes['summary_count'] = summary_counts

