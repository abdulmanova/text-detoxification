from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

# Model training parameters
args = Seq2SeqTrainingArguments(
    output_dir="models",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    report_to='tensorboard',
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Make sure to define compute_metrics
)

# Train the model
training_results = trainer.train()

# Write training results to a file
with open("training_results.txt", "w") as f:
    f.write(str(training_results))
