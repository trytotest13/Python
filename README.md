# Email Spam Detector

A simple email spam detection program using a Naive Bayes classifier. This application allows you to classify individual emails as spam or not spam based on predefined patterns, and provides an interactive interface for easy usage.

---

## Features

- **Text Preprocessing**: Removes punctuation, numbers, and unnecessary whitespace.
- **Spam Classification**: Uses Naive Bayes to classify emails as spam or not spam.
- **Interactive Menu**: Allows you to input single or multiple email messages and get spam classification.
- **Spam Detection Probability**: Outputs the likelihood of an email being spam.

---

## Requirements

- Python 3.x
- Dependencies:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `re` (built-in)
  - `string` (built-in)
  - `os` (built-in)

You can install the required libraries using:

```bash
pip install pandas numpy scikit-learn
```

---

## Getting Started

1. **Clone or Download** the repository to your local machine.

2. **Prepare the Dataset**:
   - You can replace the sample dataset in the code with your own collection of spam and ham (non-spam) emails. The dataset should be a list of text messages and labels indicating whether the message is spam (1) or not spam (0).

3. **Run the Program**:
   - Once the dataset is ready, simply run the Python script:

     ```bash
     python spam_detector.py
     ```

4. **Interactive Menu**:
   - After starting the program, you will see a menu with options:
     1. Check a single email message.
     2. Check multiple email messages at once.
     3. View common spam patterns.
     4. Exit the program.

   You can enter email texts for classification and get the result, including the probability of the message being spam.

---

## How It Works

- **Data Preprocessing**: The program first cleans the email text by:
  - Converting the text to lowercase.
  - Removing punctuation and digits.
  - Removing extra spaces.

- **Model Training**: The model is trained using the `MultinomialNB` classifier from scikit-learn. It is trained on a small sample dataset (you should replace it with your own data for better accuracy).

- **Prediction**: The model predicts whether an email is spam based on the text input. The program also provides a probability score indicating the likelihood that the email is spam.

- **Interactive Interface**: The interactive menu allows you to classify individual or multiple emails. It also educates users on common spam patterns.

---
