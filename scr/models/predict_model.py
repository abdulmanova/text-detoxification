# Model prediction parameters
prediction_args = Seq2SeqTrainingArguments(
    output_dir="model",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
)

# Trainer for prediction
prediction_trainer = Seq2SeqTrainer(
    model,
    prediction_args,
    tokenizer=tokenizer,
)

# Model prediction on a new dataset
predictions = prediction_trainer.predict(tokenized_datasets["test"])

# Write predictions to a file
with open("predictions.txt", "w") as f:
    f.write(str(predictions.predictions))
