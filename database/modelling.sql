CREATE DATABASE IF NOT EXISTS clinical_notes;

USE clinical_notes;

CREATE TABLE IF NOT EXISTS patients (
    patient_id INT, 
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    age INT,
    gender VARCHAR(10),
    contact_information  VARCHAR(255),
    emergency_contact VARCHAR(255)
);




CREATE TABLE IF NOT EXISTS clinical_notes (
    pn_num INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    case_num INT,
    pn_history TEXT
);



CREATE TABLE IF NOT EXISTS clinical_analysis_table (
    analysis_id INT AUTO_INCREMENT PRIMARY KEY,
    note_id INT,
    entity VARCHAR(255),
    text VARCHAR(255),
    date_analysis datetime
);



CREATE TABLE IF NOT EXISTS medical_conditions_table (
    condition_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    condition_name VARCHAR(255),
    status bool
);



CREATE TABLE IF NOT EXISTS medications_table (
    medication_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    medication_name VARCHAR(255),
    dosage float,
    purpose text
);


CREATE TABLE IF NOT EXISTS appointments_table (
    appointment_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    date_time datetime,
    purpose VARCHAR(255),
    notes TEXT
);



CREATE TABLE IF NOT EXISTS analysis_results_table (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id INT,
    entity  VARCHAR(255),
    entity_category VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS notes_summarization_table (
    summary_id INT AUTO_INCREMENT PRIMARY KEY,
    note_id INT,
    summary TEXT,
    chr_text INT,
    chr_sum INT
);