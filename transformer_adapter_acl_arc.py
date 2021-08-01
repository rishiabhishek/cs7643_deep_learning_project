from copy import deepcopy
import logging
import math
import sys

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets import load_dataset, ClassLabel
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaForMaskedLM,
    RobertaTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    AdamW,
    BertForMaskedLM,AutoModelForMaskedLM
)
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.adapters.configuration import AdapterConfig
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import EvalPrediction

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


data_files = {
    "train": ["Datasets/acl_arc-train.jsonl"],
    "test": ["Datasets/acl_arc-test.jsonl"],
    "validation": ["Datasets/acl_arc-dev.jsonl"],
}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
datasets = load_dataset("json", data_files=data_files).remove_columns("metadata")
labels = datasets["train"].unique("label")
labels.sort()
labels = ClassLabel(names=labels)


def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )
    tokens["label"] = labels.str2int(examples["label"])
    return tokens


tokenized_datasets = datasets.map(tokenize_function, batched=True, load_from_cache_file=False)

##############################
# setting learning rate

learning_rate = 1e-4
adatapter_type = "houlsby"
##############################


def run_mlm():
    logger.info("Running Masked Language Modeling fine-tuning (Domain Adaptation)")

    tokenized_mlm_dataset = tokenized_datasets.remove_columns("label")

    # Using pfeiffer adapter config for training
    # adapter_config = AdapterConfig.load("houlsby", reduction_factor=16)
    adapter_config = AdapterConfig.load(adatapter_type, reduction_factor=16)

    logger.info("Loading roberta-base..................")
    lm_model = RobertaForMaskedLM.from_pretrained("roberta-base")
    
    # adding adapter to the model
    lm_model.add_adapter("acl_arc", config=adapter_config)

    # # Freezing all model parameters except adapters
    lm_model.train_adapter("acl_arc")

    # # Set the adapters to be used in every forward pass
    lm_model.set_active_adapters("acl_arc")

    trainable_prams = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)

    print(f'Trainable parameter : {trainable_prams}')

    lm_training_args = TrainingArguments(
        run_name="mlm_alc_arc",
        output_dir=f"training_output_adapters/acl_arc_{learning_rate}/language_modeling/",
        do_train=True,
        num_train_epochs=50,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=3,
        log_level="debug",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15,
    )
    lm_trainer = Trainer(
        model=lm_model,
        args=lm_training_args,
        data_collator=data_collator,
        train_dataset=tokenized_mlm_dataset["train"],
        eval_dataset=tokenized_mlm_dataset["validation"],
        do_save_full_model=False,
        do_save_adapters=True,
        callbacks=[early_stopping],
    )

    # Training
    train_result = lm_trainer.train()
    lm_trainer.save_model()  # Saves the tokenizer too for easy upload
    train_metrics = train_result.metrics
    lm_trainer.log_metrics("train", train_metrics)

    # Testing
    test_metrics = lm_trainer.evaluate(eval_dataset=tokenized_mlm_dataset["test"])
    try:
        perplexity = math.exp(test_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    test_metrics["perplexity"] = perplexity
    lm_trainer.log_metrics("eval", test_metrics)


# Task Adaptation
def run_classification():
    logger.info("Running Classification Task (Task Adaptation)")

    tokenized_task_dataset = deepcopy(tokenized_datasets)

    #  Transform to pytorch tensors and only output the required columns
    tokenized_task_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Loding Roberta model
    roberta_config = RobertaConfig.from_pretrained(
        "roberta-base", num_labels=labels.num_classes
    )
    task_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=roberta_config
    )

    # Load previously trained Adapter
    task_model.load_adapter(f"training_output_adapters/acl_arc_{learning_rate}/language_modeling/acl_arc")

    # Freezing all model parameters except adapters
    task_model.train_adapter("acl_arc")

    # Set the adapters to be used in every forward pass
    task_model.set_active_adapters("acl_arc")

    task_training_args = TrainingArguments(
        run_name="task_alc_arc",
        output_dir=f"training_output_adapters/acl_arc_{learning_rate}/classification/",
        do_train=True,
        num_train_epochs=50,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=5,
        log_level="debug",
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model='f1',
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
    tensorboard = TensorBoardCallback()

    task_trainer = Trainer(
        model=task_model,
        args=task_training_args,
        train_dataset=tokenized_task_dataset["train"],
        eval_dataset=tokenized_task_dataset["validation"],
        do_save_full_model=False,
        do_save_adapters=True,
        callbacks=[early_stopping, tensorboard],
        compute_metrics=compute_metrics
    )

    logger.info(tokenized_task_dataset["train"].column_names)
    logger.info(f"Number of trainable parameters : {sum(p.numel() for p in task_model.parameters() if p.requires_grad)}")
    # Training
    train_result = task_trainer.train()
    # task_trainer.save_model()  # Saves the tokenizer too for easy upload
    train_metrics = train_result.metrics
    task_trainer.log_metrics("train", train_metrics)

    # Testing
    test_metrics = task_trainer.evaluate(eval_dataset=tokenized_task_dataset["test"])
    task_trainer.log_metrics("eval", test_metrics)

    task_trainer.model.save_adapter(f"training_output_adapters/acl_arc_{learning_rate}/classification/", "acl_arc", with_head=True)


def main(args):
    if args == 'mlm':
        run_mlm()
    elif args == 'classification':
        run_classification()
    else:
        run_mlm()
        run_classification()


if __name__ == "__main__":
    args = sys.argv
    args.append('mlm')
    print(args)
    main(args[1])
