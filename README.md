# T5-Based Detoxification Model

This is a T5-based model fine-tuned for detoxifying toxic text.

## Prerequisites

- Python 3.x
- Install required packages: `pip install -r requirements.txt`

## Usage
Train the Model
`python train.py --model_name t5-base --epochs 5 --batch_size 32`

Run Inference
`python predict.py --input_file input.txt --output_file output.txt`

Concurrent Execution
For concurrent execution, you can use the & operator in Unix-like systems.
`python train.py --model_name t5-base --epochs 5 --batch_size 32 & python predict.py --input_file input.txt --output_file output.txt`
This will run both commands concurrently.

Contributors
Abdulmanova Alsu (BS21-RO-01)