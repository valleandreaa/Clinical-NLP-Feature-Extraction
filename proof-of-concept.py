import torch
from transformers import pipeline

# Load a pre-trained NER model (e.g., "dbmdz/bert-large-cased-finetuned-conll03-english")
nlp_ner = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner")
#TODO: find preprocess

# Sample clinical text
clinical_text = "17-year-old male, has come to the student health clinic complaining of heart pounding. Mr. Cleveland's mother has given verbal consent for a history, physical examination, and treatment-began 2-3 months ago,sudden,intermittent for 2 days(lasting 3-4 min),worsening,non-allev/aggrav-associated with dispnea on exersion and rest,stressed out about school-reports fe feels like his heart is jumping out of his chest-ros:denies chest pain,dyaphoresis,wt loss,chills,fever,nausea,vomiting,pedal edeam-pmh:non,meds :aderol (from a friend),nkda-fh:father had MI recently,mother has thyroid dz-sh:non-smoker,mariguana 5-6 months ago,3 beers on the weekend, basketball at school-sh:no std"

# Apply the NER model
ner_results = nlp_ner(clinical_text)

# Extract and print named entities
for entity in ner_results:
    print(f"Entity: {entity['word']} - Label: {entity['entity']}")



# You can filter entities based on labels relevant to your task (e.g., 'MEDICATION', 'DATE', 'CONDITION', etc.).


# # Fine-Tuning Parameters
# training_args = TrainingArguments(
#     output_dir="./ner_finetuned",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=32,
#     save_steps=10_000,
#     save_total_limit=2,
# )
#
# # Fine-Tuning Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
# )

# Fine-Tuning
# You need to prepare your fine-tuning dataset and data collator
# Refer to the Transformers library documentation for details

# Train the model
# trainer.train()

# Save the fine-tuned model
# trainer.save_model()

# Load the fine-tuned model
# model = BertForTokenClassification.from_pretrained("./ner_finetuned")

# Apply the fine-tuned NER model
# nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)

# Sample text with clinical and date information
# text = "The patient was prescribed aspirin for hypertension on October 15, 2023."

# ner_results = nlp_ner(text)
# for entity in ner_results:
#     print(f"Entity: {entity['word']} - Label: {entity['entity']}")