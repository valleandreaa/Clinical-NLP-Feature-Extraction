-- Create index on patient_id column in patients table
CREATE INDEX idx_patient_id ON patients(patient_id);

-- Add foreign key constraint for patient_id column in clinical_notes table
ALTER TABLE clinical_notes
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

-- Add foreign key constraint for note_id column in clinical_analysis_table table
ALTER TABLE clinical_analysis_table
ADD FOREIGN KEY (note_id) REFERENCES clinical_notes(pn_num);

-- Add foreign key constraint for patient_id column in medical_conditions_table table
ALTER TABLE medical_conditions_table
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

-- Add foreign key constraint for patient_id column in medications_table table
ALTER TABLE medications_table
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

-- Add foreign key constraint for patient_id column in appointments_table table
ALTER TABLE appointments_table
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

-- Add foreign key constraint for analysis_id column in analysis_results_table table
ALTER TABLE analysis_results_table
ADD FOREIGN KEY (analysis_id) REFERENCES clinical_analysis_table(analysis_id);

-- Add foreign key constraint for note_id column in notes_summarization_table table
ALTER TABLE notes_summarization_table
ADD FOREIGN KEY (note_id) REFERENCES clinical_notes(pn_num);
