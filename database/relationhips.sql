-- Add an index to the patient_id column in the patients table
CREATE INDEX idx_patient_id ON patients(patient_id);

ALTER TABLE features
ADD FOREIGN KEY (pn_num) REFERENCES patients(patient_id);

ALTER TABLE  features
ADD FOREIGN KEY (pn_num) REFERENCES patient_notes(pn_num);
