SET GLOBAL local_infile = true;
USE clinical_notes;


LOAD DATA LOCAL INFILE 'C:\\Users\\andreavalle\\Desktop\\clinical-notes\\datasets\\patient_notes.csv' 
INTO TABLE patient_notes
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(pn_num, case_num, pn_history);

LOAD DATA LOCAL INFILE 'C:\\Users\\andreavalle\\Desktop\\clinical-notes\\datasets\\patients.csv' 
INTO TABLE patients
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(patient_id,first_name,last_name);

SET SQL_SAFE_UPDATES = 0;
UPDATE clinical_notes
SET case_num = patient_id;
SET SQL_SAFE_UPDATES = 1;
