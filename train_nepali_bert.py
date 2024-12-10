import torch

import logging
import sys
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import optuna
import matplotlib.pyplot as plt


ENTITY_TYPES = [
    "O",
    "B-Person",
    "I-Person",
    "B-Organization",
    "I-Organization",
    "B-Location",
    "I-Location",
    "B-Date",
    "I-Date",
    "B-Event",
    "I-Event",
]


def prepare_dataset(file_name):

    sentences = []
    tags = []
    full_sentences = []

    with open(file_name, "r") as f:

        current_sentence = []
        current_tags = []

        for line in f:
            line = line.strip()

            # Check for empty line, indicating the end of a sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    full_sentences.append(" ".join(current_sentence))
                    current_sentence = []
                    current_tags = []
            else:
                word, tag = line.split()
                current_sentence.append(word)
                current_tags.append(ENTITY_TYPES.index(tag))

        # Append the last sentence if the file does not end with a newline
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)

    formatted_data = {"sentence": sentences, "tags": tags}
    return Dataset.from_dict(formatted_data)


def subset_to_huggingface_dataset(subset):
    # Extract data from the subset
    data = [subset.dataset[idx] for idx in subset.indices]

    # Convert the list of dictionaries to a Hugging Face Dataset
    return Dataset.from_list(data)


train_dataset = torch.load("train_dataset.pt")
validation_dataset = torch.load("validation_dataset.pt")
test_dataset = torch.load("test_dataset.pt")


ner_datasets = DatasetDict(
    {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }
)


def tokenize_and_align_tags(samples):
    tokenized_inputs = tokenizer(
        samples["sentence"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(samples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# nep_model = AutoModelForTokenClassification.from_pretrained(
#     "NepBERTa/NepBERTa", from_tf=True, num_labels=len(ENTITY_TYPES)
# )
# nep_tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa", model_max_length=512)


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ENTITY_TYPES[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ENTITY_TYPES[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "classification_report": classification_report(true_labels, true_predictions),
    }


def data_collator(data):
    input_ids = [torch.tensor(item["input_ids"]) for item in data]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
    labels = [torch.tensor(item["labels"]) for item in data]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return data


def main(model, tokenized_datasets, n_epochs, learning_rate, result_dir):

    training_args = TrainingArguments(
        output_dir=result_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Using current device:", training_args.device)

    trainer.train()


def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [3, 5, 7])
    eval_batch_size = trial.suggest_categorical("eval_batch_size", [3, 5, 7])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)

    result_dir = "./saved_model_multilingual"

    training_args = TrainingArguments(
        output_dir=result_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=batch_size,
        logging_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        run_name=f"{result_dir}_{batch_size}_{eval_batch_size}_{learning_rate}_{n_epochs}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Using current device:", training_args.device)

    trainer.train()
    eval_results = trainer.evaluate()

    return eval_results["eval_loss"]


multi_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
multi_model = BertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=len(ENTITY_TYPES)
)

tokenizer = multi_tokenizer
model = multi_model

tokenized_datasets = ner_datasets.map(tokenize_and_align_tags, batched=True)


if __name__ == "__main__":
    log_filename = "multinlingual_optuna_study.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    # optuna.logging.disable_default_handler()

    tokenizer = multi_tokenizer
    model = multi_model
    tokenized_datasets = ner_datasets.map(tokenize_and_align_tags, batched=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, 20)

    best_trial = study.best_trial
    print("*" * 10)
    print("Best Trial Number:", best_trial.number)
    print("Lowest Loss:", best_trial.value)
    print("Best Hyperparameters:", best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    plt.savefig("multilingual_optuna_optimization_history.png")
    # main(model, tokenized_datasets, 5, 5e-5, "./saved_model_nepali")
