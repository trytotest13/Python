Email Spam Detector
A simple email spam detection program using a Naive Bayes classifier. This application allows you to classify individual emails as spam or not spam based on predefined patterns, and provides an interactive interface for easy usage.

Features
Text Preprocessing: Removes punctuation, numbers, and unnecessary whitespace.
Spam Classification: Uses Naive Bayes to classify emails as spam or not spam.
Interactive Menu: Allows you to input single or multiple email messages and get spam classification.
Spam Detection Probability: Outputs the likelihood of an email being spam.
Requirements
Python 3.x
Dependencies:
pandas
numpy
sklearn
re (built-in)
string (built-in)
os (built-in)
You can install the required libraries using:

bash
Copy code
pip install pandas numpy scikit-learn
Getting Started
Clone or Download the repository to your local machine.

Prepare the Dataset:

You can replace the sample dataset in the code with your own collection of spam and ham (non-spam) emails. The dataset should be a list of text messages and labels indicating whether the message is spam (1) or not spam (0).
Run the Program:

Once the dataset is ready, simply run the Python script:

bash
Copy code
python spam_detector.py
Interactive Menu:

After starting the program, you will see a menu with options:
Check a single email message.
Check multiple email messages at once.
View common spam patterns.
Exit the program.
You can enter email texts for classification and get the result, including the probability of the message being spam.

How It Works
Data Preprocessing: The program first cleans the email text by:

Converting the text to lowercase.
Removing punctuation and digits.
Removing extra spaces.
Model Training: The model is trained using the MultinomialNB classifier from scikit-learn. It is trained on a small sample dataset (you should replace it with your own data for better accuracy).

Prediction: The model predicts whether an email is spam based on the text input. The program also provides a probability score indicating the likelihood that the email is spam.

Interactive Interface: The interactive menu allows you to classify individual or multiple emails. It also educates users on common spam patterns.

Example Output
bash
Copy code
Email text: Congratulations! You won $1,000,000! Claim now!
Classification: SPAM
Spam Probability: 95.00%
🚫 High Risk - Be Cautious!
Usage
Check Single Email: Input one email at a time, and the program will output whether it is spam or not, along with the spam probability.

Check Multiple Emails: You can input multiple emails and get a classification result for each one.

View Sample Spam Patterns: The program displays common spam patterns to watch out for, such as:

Excessive use of capital letters.
Urgency words like "Act now" and "Limited time".
Requests for personal information.
Troubleshooting
If you encounter errors during execution, ensure that the required libraries (pandas, numpy, sklearn) are installed properly.
If the program crashes or freezes, check the dataset format to ensure it matches the expected structure.
License
This project is licensed under the MIT License.

Future Improvements
Integrating more sophisticated preprocessing techniques like stemming or lemmatization.
Using a larger and more diverse dataset for better classification accuracy.
Adding an option to save and load trained models for reuse.
Note:
Make sure to adjust the train_model() method with your own dataset for optimal results. The current dataset is a small sample for illustration purposes.
