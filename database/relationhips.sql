-- Add an index to the patient_id column in the patients table
CREATE INDEX idx_patient_id ON patients(patient_id);

ALTER TABLE clinical_notes
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

ALTER TABLE clinical_analysis_table
ADD FOREIGN KEY (note_id) REFERENCES clinical_notes(pn_num);

ALTER TABLE medical_conditions_table
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

ALTER TABLE medications_table
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

ALTER TABLE appointments_table
ADD FOREIGN KEY (patient_id) REFERENCES patients(patient_id);

ALTER TABLE analysis_results_table
ADD FOREIGN KEY (analysis_id) REFERENCES clinical_analysis_table(analysis_id);
