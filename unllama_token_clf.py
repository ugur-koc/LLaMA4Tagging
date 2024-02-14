# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import evaluate
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

from modeling_llama import UnmaskingLlamaForTokenClassification

DED_NER_LABELS = [
    "O", "B-None", "I-None",
    "B-AccountData.AccountID", "I-AccountData.AccountID",
    "B-AccountData.AccountName", "I-AccountData.AccountName",
    "B-AccountData.AccountPassword", "I-AccountData.AccountPassword",
    "B-AccountData.LanguagePreferences", "I-AccountData.LanguagePreferences",
    "B-BackgroundCheckDetails.EmploymentHistory", "I-BackgroundCheckDetails.EmploymentHistory",
    "B-ContactData.Address", "I-ContactData.Address",
    "B-ContactData.EmailAddress", "I-ContactData.EmailAddress",
    "B-ContactData.EmergencyContactDetails", "I-ContactData.EmergencyContactDetails",
    "B-ContactData.PhoneNumber", "I-ContactData.PhoneNumber",
    "B-EducationalBackground.EducationalHistory", "I-EducationalBackground.EducationalHistory",
    "B-FinancialData.BankAccountDetails", "I-FinancialData.BankAccountDetails",
    "B-FinancialData.CardIssuer", "I-FinancialData.CardIssuer",
    "B-FinancialData.CardNumber", "I-FinancialData.CardNumber",
    "B-FinancialData.InsuranceNumber", "I-FinancialData.InsuranceNumber",
    "B-FinancialData.PaymentMode", "I-FinancialData.PaymentMode",
    "B-FinancialData.PayrollInformation", "I-FinancialData.PayrollInformation",
    "B-FinancialData.Salary", "I-FinancialData.Salary",
    "B-FinancialData.TaxInformation", "I-FinancialData.TaxInformation",
    "B-HealthData.DisabilityorSpecificCondition", "I-HealthData.DisabilityorSpecificCondition",
    "B-HealthData.IllnessorMedicalCondition", "I-HealthData.IllnessorMedicalCondition",
    "B-LocationData.PreciseLocation", "I-LocationData.PreciseLocation",
    "B-NationalIdentificationNumbers.DrivingLicense", "I-NationalIdentificationNumbers.DrivingLicense",
    "B-OnlineIdentifiers.AdvertisingIdentifiers", "I-OnlineIdentifiers.AdvertisingIdentifiers",
    "B-OnlineIdentifiers.Cookies", "I-OnlineIdentifiers.Cookies",
    "B-OnlineIdentifiers.DeviceIdentifier", "I-OnlineIdentifiers.DeviceIdentifier",
    "B-OnlineIdentifiers.IPAddress", "I-OnlineIdentifiers.IPAddress",
    "B-OnlineIdentifiers.Pixel", "I-OnlineIdentifiers.Pixel",
    "B-PersonalCharacteristics.Age", "I-PersonalCharacteristics.Age",
    "B-PersonalCharacteristics.Height", "I-PersonalCharacteristics.Height",
    "B-PersonalCharacteristics.Nationality", "I-PersonalCharacteristics.Nationality",
    "B-PersonalCharacteristics.RacialorEthnicOrigin", "I-PersonalCharacteristics.RacialorEthnicOrigin",
    "B-PersonalCharacteristics.Religion/ReligiousBeliefs", "I-PersonalCharacteristics.Religion/ReligiousBeliefs",
    "B-PersonalCharacteristics.Signature", "I-PersonalCharacteristics.Signature",
    "B-PersonalCharacteristics.Weight", "I-PersonalCharacteristics.Weight",
    "B-PersonalIdentification.Age", "I-PersonalIdentification.Age",
    "B-PersonalIdentification.DateofBirth", "I-PersonalIdentification.DateofBirth",
    "B-PersonalIdentification.EmployeeCode", "I-PersonalIdentification.EmployeeCode",
    "B-PersonalIdentification.FirstName", "I-PersonalIdentification.FirstName",
    "B-PersonalIdentification.Gender", "I-PersonalIdentification.Gender",
    "B-PersonalIdentification.LastName", "I-PersonalIdentification.LastName",
    "B-PersonalIdentification.Photograph", "I-PersonalIdentification.Photograph",
    "B-ProfessionalAndEmploymentBackground.TotalExperience", "I-ProfessionalAndEmploymentBackground.TotalExperience",
    "B-PurchaseData.OfferDetails", "I-PurchaseData.OfferDetails",
    "B-PurchaseData.OrderDetails", "I-PurchaseData.OrderDetails",
    "B-PurchaseData.ProductReturnHistory", "I-PurchaseData.ProductReturnHistory",
    "B-PurchaseData.PurchaseHistory", "I-PurchaseData.PurchaseHistory",
    "B-SocialMediaData.SocialMediaAccount", "I-SocialMediaData.SocialMediaAccount",
    "B-SpouseFamilyAndDependentDetails.ParentsName", "I-SpouseFamilyAndDependentDetails.ParentsName",
    "B-TechnicalData.BrowsingHistory", "I-TechnicalData.BrowsingHistory",
    "B-TechnicalData.TechnicalDiagnosticData", "I-TechnicalData.TechnicalDiagnosticData",
    "B-UsageData.ClickStream", "I-UsageData.ClickStream",
    "B-UserContentData.EmailsorTextMessages", "I-UserContentData.EmailsorTextMessages",
    "B-UserContentData.Ratings", "I-UserContentData.Ratings",
    "B-UserContentData.Reviews", "I-UserContentData.Reviews",
    "B-WorkplaceMonitoringData.ApplicationUsage", "I-WorkplaceMonitoringData.ApplicationUsage"
]

def load_ontonotesv5():
    ret = {}
    for split_name in ['train', 'dev', 'test']:
        data = []
        with open(f'./data/ontonotesv5/{split_name}.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)


if len(sys.argv) != 3:
    print('usage python %.py task model_size')
    sys.exit()

task, model_size = sys.argv[1], sys.argv[2].lower()
print(f'handling task {task}')

epochs = 10
batch_size = 8
learning_rate = 1e-4
max_length = 4096
lora_r = 12
if model_size == '7b':
    model_id = 'NousResearch/Llama-2-7b-hf'
elif model_size == '13b':
    model_id = 'NousResearch/Llama-2-13b-hf'
else:
    raise NotImplementedError
tokenizer = AutoTokenizer.from_pretrained(model_id)
seqeval = evaluate.load("seqeval")
if task == 'wnut_17':
    ds = load_dataset("wnut_17")
    label2id = { "O": 0, "B-corporation": 1, "I-corporation": 2, "B-creative-work": 3, "I-creative-work": 4, "B-group": 5, "I-group": 6, "B-location": 7, "I-location": 8, "B-person": 9, "I-person": 10, "B-product": 11, "I-product": 12, }
elif task == 'conll2003':
    ds = load_dataset("conll2003")
    label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
elif task == 'ontonotesv5':
    ds = load_ontonotesv5()
    label2id = {'O': 0, 'B-NORP': 1, 'B-PERSON': 2, 'B-WORK_OF_ART': 3, 'B-QUANTITY': 4, 'B-EVENT': 5, 'B-DATE': 6, 'B-TIME': 7, 'B-PERCENT': 8, 'B-LANGUAGE': 9, 'B-ORG': 10, 'B-CARDINAL': 11, 'B-LAW': 12, 'B-GPE': 13, 'B-PRODUCT': 14, 'B-LOC': 15, 'B-MONEY': 16, 'B-ORDINAL': 17, 'B-FAC': 18}
elif task == 'ded':
    ds = load_dataset("json",
                      data_files={"train": "./data/final_labelled_data_ner.jsonl"},
                      split="train").train_test_split(test_size=0.2)
    label2id = {label: i for i, label in enumerate(DED_NER_LABELS)}
else:
    raise NotImplementedError

id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys()) # ds["train"].features[f"ner_tags"].feature.names
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=torch.float16  # Computation type
)
model = UnmaskingLlamaForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id, device_map="auto", quantization_config=bnb_config
)
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=lora_r, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="my_awesome_ds_model",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
