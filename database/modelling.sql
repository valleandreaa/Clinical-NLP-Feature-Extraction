CREATE DATABASE IF NOT EXISTS clinical_notes;

USE clinical_notes;



CREATE TABLE IF NOT EXISTS patients (
    patient_id INT, 
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    age INT,
    gender VARCHAR(10),
    problem TEXT,
    treatment TEXT, 
    disease TEXT
);


CREATE TABLE IF NOT EXISTS features (
    feature_num INT  PRIMARY KEY,
    pn_num INT,
    feature_text TEXT, 
    feature_type TEXT
);


CREATE TABLE IF NOT EXISTS patient_notes (
    pn_num INT AUTO_INCREMENT PRIMARY KEY,
    case_num INT,
    pn_history TEXT    
);


