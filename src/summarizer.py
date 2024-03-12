import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from database import *

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")

def apply_summarizer(text):
    """
    Apply summarizer model to individual texts.
    """
    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="pt")
    summaries = model.generate(input_ids, max_length=230, min_length=30, do_sample=False)
    return tokenizer.decode(summaries[0], skip_special_tokens=True)

def main():
    try:
        # Establish connection
        conn = connect_to_database()
        if conn:
            cursor = conn.cursor()

            # Fetch data from clinical_notes
            query = "SELECT pn_num, clinical_text FROM clinical_notes"
            cursor.execute(query)
            rows = cursor.fetchall()

            # Process each row
            for pn_num, clinical_text in rows:
                summary = apply_summarizer(clinical_text)

                # Insert the summary without character counts
                query_insert = "INSERT INTO notes_summarization_table (note_id, summary) VALUES (%s, %s)"
                cursor.execute(query_insert, (pn_num, summary))

                # Update the record with character counts
                query_update = """
                UPDATE notes_summarization_table
                SET chr_text = LENGTH(%s),
                    chr_sum = LENGTH(%s)
                WHERE note_id = %s
                """
                cursor.execute(query_update, (clinical_text, summary, pn_num))
                conn.commit()

            print("Summarization completed successfully.")

    except Exception as e:
        print(f"Error during summarization: {e}")
    finally:
        # Close the cursor and connection
        close_connection(conn, cursor)

if __name__ == "__main__":
    main()
