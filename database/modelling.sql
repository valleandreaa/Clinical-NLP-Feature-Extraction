-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS clinical_notes;

-- Use the clinical_notes database
USE clinical_notes;

-- Table for storing patient information
CREATE TABLE IF NOT EXISTS patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    age TINYINT UNSIGNED,
    gender ENUM('Male', 'Female', 'Other'),
    contact_information VARCHAR(255),
    emergency_contact VARCHAR(255)
);

-- Table for storing clinical notes
CREATE TABLE IF NOT EXISTS clinical_notes (
    pn_num INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    case_num INT,
    pn_history TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Table for storing clinical analysis results
CREATE TABLE IF NOT EXISTS clinical_analysis_table (
    analysis_id INT AUTO_INCREMENT PRIMARY KEY,
    note_id INT,
    entity VARCHAR(255),
    text TEXT,
    date_analysis DATETIME,
    FOREIGN KEY (note_id) REFERENCES clinical_notes(pn_num)
);

-- Table for storing medical conditions
CREATE TABLE IF NOT EXISTS medical_conditions_table (
    condition_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    condition_name VARCHAR(255),
    status BOOLEAN,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Table for storing medications
CREATE TABLE IF NOT EXISTS medications_table (
    medication_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    medication_name VARCHAR(255),
    dosage VARCHAR(50),
    purpose TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Table for storing appointments
CREATE TABLE IF NOT EXISTS appointments_table (
    appointment_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    date_time DATETIME,
    purpose VARCHAR(255),
    notes TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Table for storing summarized clinical notes
CREATE TABLE IF NOT EXISTS notes_summarization_table (
    summary_id INT AUTO_INCREMENT PRIMARY KEY,
    note_id INT,
    summary TEXT,
    chr_text INT,
    chr_sum INT,
    FOREIGN KEY (note_id) REFERENCES clinical_notes(pn_num)
);
