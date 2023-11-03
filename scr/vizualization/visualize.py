import matplotlib.pyplot as plt

# Read evaluation results from file
with open("training_results.txt", "r") as f:
    evaluation_results = eval(f.read())

# Bar plot of evaluation metrics
fig, ax = plt.subplots()
ax.bar(evaluation_results.keys(), evaluation_results.values())
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics')

plt.show()
