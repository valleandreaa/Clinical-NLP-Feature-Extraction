from transformers import pipeline
from database  import *

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

# Insert data into medical_conditions_table
insert_query = """
INSERT INTO medical_conditions_table (patient_id, condition_name, status)
VALUES (%s, %s, %s)
"""

# Iterate over the results and insert each into the medical_conditions_table
for patient_id, condition_text in rows:
    status = True
    cursor.execute(insert_query, (patient_id, condition_text, status))

# Commit the changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()