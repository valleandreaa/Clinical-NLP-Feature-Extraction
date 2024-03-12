# Clinical Notes Database Management (DBM)

## Introduction & Initial Situation

The journey towards complete digitization in the Swiss healthcare system is an ongoing process, with hospitals playing a pivotal role in its realization. As artificial intelligence (AI) gains traction in healthcare, natural language processing (NLP) models emerge as potential catalysts for addressing the digitization challenge. NLP, a subset of AI, enables the extraction of complex clinical data by converting free-form text into a standardized format.

In this project, we assess the feasibility of integrating NLP capabilities with relational databases to develop a system capable of processing and storing clinical notes. Specifically, we leverage two specialized NLP subfields and demonstrate their integration into a database framework. Firstly, Named Entity Recognition (NER) utilizing a ClinicalBERT transformer to extract entities such as problems, medications, and conditions from clinical notes. Secondly, we employ a T5 (Text-to-Text Transfer Transformer) for summarizing the content of these notes. This approach enables efficient handling and organization of complex clinical data.

## Objective

Our objective is to model, load, transform, query, and optimize a medical clinical notes database integrated with the existing electronic health record (EHR) system for the University of Zurich Hospital (USZ). This proof of concept, if successful, could potentially expand into a continued partnership to further digitize other areas within the hospital. Through this project, we aim to gain experience in using SQL database technologies and applying them to practical use cases. The project outcome will aid in making informed decisions through data visualizations using a Business Intelligence (BI) tool. We utilize MySQL as our SQL database and Metabase as our visualization tool, with the flexibility to explore alternative products within the same technology category if necessary. Ultimately, our goal is to provide a dashboard enabling clinicians to access accurate and up-to-date clinical information on their patients.

## Project Idea and Use Case

### Motivation

The project is motivated by Switzerland's healthcare sector's ongoing digital transformation efforts and the need to address unstructured clinical notes. Research indicates that a significant portion of healthcare data, approximately 80%, remains unstructured, underscoring the need for innovative solutions to leverage this data effectively.

### Decision Support

Clinicians face challenges such as excessive time spent on electronic health record (EHR) documentation, impacting patient care, and the difficulty of identifying key features from clinical notes. Our project aims to alleviate these pain points by streamlining clinical note processing and extracting essential information for decision-making.

### Data

We obtained our dataset from Kaggle, provided by the National Board of Medical Examiners (NBME) for their competition "NBME - Score Clinical Patient Notes." This dataset, originating from the USMLE® Step 2 Clinical Skills examination, comprises patient notes containing various features. We chose this dataset due to its open-source nature and similarity to clinical notes taken at USZ. Additionally, we generated synthetic data to augment our dataset.

### Database Technology

- **Purpose**: The database serves to efficiently store, organize, and manage clinical notes and patient data, leveraging NLP transformers for information extraction.
- **Database Software**: MySQL is chosen for its robustness, scalability, and strong community support, suitable for healthcare applications.
- **Elements and Data Flows**: The system includes data input interfaces, storage mechanisms, query processing modules, and output interfaces for reporting. Data flows involve inputting clinical data, processing it for storage, and retrieving it for analysis and reporting.
- **SQL or NoSQL**: SQL database (MySQL) is preferred for its structured query language and reliability in handling structured data.
- **Implementation and Integration**: The database is implemented as part of a larger healthcare information system and integrated with existing EHR systems for seamless data exchange.
- **Data Entry and Migration**: Data is entered through automated extraction tools and manual entry interfaces, with migration from existing systems facilitated by ETL processes.
- **Querying, Manipulation, and Transformation**: SQL queries are used for data retrieval, manipulation, and transformation, with custom scripts employed for specific analytical purposes.
- **Optimization**: Indexing critical fields, optimizing query structures, and using caching mechanisms are key optimization strategies to manage volume and enhance speed.

## Conclusion

In conclusion, the "DBM - Clinical Notes" project represents a comprehensive exploration of database management and clinical data analysis. The project successfully developed a robust database system leveraging NLP techniques for managing clinical notes efficiently. Integration with advanced visualization tools enhances the accessibility and utility of clinical data, empowering stakeholders to make informed decisions. This project sets a precedent for future endeavors in healthcare data management and analysis, emphasizing the importance of continuous learning and adaptation in the evolving landscape of data science and healthcare technology.