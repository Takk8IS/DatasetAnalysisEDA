# Documentation of Measures and Visualisations 📊

## Introduction

This section details the main measures used in the dataset analysis, providing an in-depth explanation of each metric and its relevance in the context of the study. Additionally, we present visualisations in the form of histograms for a more intuitive understanding of the data distribution.

## Analysed Measures

### 1. Precision

Precision is a crucial metric in classification models, representing the proportion of correct positive predictions in relation to the total positive predictions made. In the context of this study, precision indicates the model's ability to correctly identify cases of housing loss.

A high precision value means that when the model predicts a student has lost housing, this prediction is generally correct. This is particularly important in situations where false positives can have significant consequences, such as unnecessary alerts or inadequate resource allocation.

### 2. Recall

Recall, also known as sensitivity, measures the proportion of actual positive cases that were correctly identified by the model. In our study, recall indicates the model's ability to correctly identify all cases of housing loss.

A high recall value is crucial when it is important not to miss any positive cases, even if it means some false positives. In the context of student housing loss, high recall ensures that most at-risk students are identified, allowing for early interventions.

### 3. F1 Score

The F1 score is the harmonic mean between precision and recall, providing a single value that balances both metrics. This measure is particularly useful when seeking a balance between precision and recall, especially in imbalanced datasets.

In the context of predicting housing loss, a high F1 Score indicates that the model is capable of identifying risk cases with high precision, without failing to capture the majority of actual cases. This is ideal for early warning systems, where both false positives and false negatives can have significant implications.

## Visualisations

For each of these measures, we use histograms to visualise their distribution. Histograms offer a graphical representation of the frequency of different values in a dataset, allowing for a quick understanding of the distribution and identification of patterns or anomalies.

### Code for Histogram Generation

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()

# Example of use
plot_histogram(precision_scores, 'Precision Distribution', 'Precision', 'Frequency')
plot_histogram(recall_scores, 'Recall Distribution', 'Recall', 'Frequency')
plot_histogram(f1_scores, 'F1 Score Distribution', 'F1 Score', 'Frequency')
```

### Interpretation of Histograms

1. **Precision Histogram**: Shows the distribution of precision values. A histogram concentrated on higher values indicates a model with good positive prediction capability.

2. **Recall Histogram**: Illustrates the distribution of recall values. A concentration on high values suggests that the model is effective in identifying the majority of actual positive cases.

3. **F1 Score Histogram**: Represents the balance between precision and recall. A histogram with a peak at high values indicates good overall model performance.

These visualisations, combined with detailed explanations of each metric, provide a comprehensive understanding of the model's performance and the distribution of key measures in the analysed dataset.

## Donations

If this project was useful, consider making a donation:

**USDT (TRC-20)**: `TGpiWetnYK2VQpxNGPR27D9vfM6Mei5vNA`

Your support helps us continue developing innovative data analysis tools.

## Licence

This project is licensed under the CC-BY-4.0 Licence.

## About Takk™ Innovate Studio

Leading the Digital Revolution as the 100% Artificial Intelligence Pioneer Team.

-   Author: [David C Cavalcante](mailto:davcavalcante@proton.me)
-   LinkedIn: [linkedin.com/in/hellodav](https://www.linkedin.com/in/hellodav/)
-   X: [@Takk8IS](https://twitter.com/takk8is/)
-   Medium: [takk8is.medium.com](https://takk8is.medium.com/)
-   Website: [takk.ag](https://takk.ag/)
