import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import re
import string
import time
import os

class SpamDetector:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        
    def preprocess_text(self, text):
        """Preprocess the email text by lowering case, removing punctuation and digits."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text
    
    def train(self, X_train, y_train):
        """Train the spam detector model."""
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        self.classifier.fit(X_train_vectorized, y_train)
    
    def predict(self, text):
        """Predict whether a single email is spam or not."""
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(vectorized_text)
        probability = self.classifier.predict_proba(vectorized_text)
        
        return {
            'is_spam': bool(prediction[0]),
            'spam_probability': probability[0][1]
        }

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header."""
    print("=" * 50)
    print("           EMAIL SPAM DETECTOR")
    print("=" * 50)
    print()

def print_result(text, result):
    """Print the spam detection result in a formatted way."""
    print("\nANALYSIS RESULT:")
    print("-" * 30)
    print(f"Email text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Classification: {'SPAM' if result['is_spam'] else 'NOT SPAM'}")
    print(f"Spam Probability: {result['spam_probability']:.2%}")
    
    if result['is_spam']:
        print("🚫 High Risk - Be Cautious!")
    else:
        print("✅ Looks Safe!")
    print("-" * 30)

def train_model():
    """Train the model with sample data."""
    data = {
        'text': [
            'CONGRATULATIONS! You won $1,000,000! Claim now!',
            'Meeting scheduled for tomorrow at 10 AM',
            'Get rich quick! Work from home!',
            'Your package has been delivered',
            'FREE VIAGRA! Best prices guaranteed!!!',
            'Project deadline reminder: reports due Friday',
            'Limited time offer! Act now!',
            'Please review the attached document',
            'Buy now! Discount 90% off!',
            'Your account statement is ready'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for ham
    }
    
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    detector = SpamDetector()
    detector.train(X_train, y_train)

    # Evaluate model performance on test set
    X_test_processed = [detector.preprocess_text(text) for text in X_test]
    X_test_vectorized = detector.vectorizer.transform(X_test_processed)
    
    predictions = detector.classifier.predict(X_test_vectorized)
    
    # Print classification report
    print("\nModel Evaluation:")
    print(classification_report(y_test, predictions))
    
    return detector

def interactive_menu():
    """Display and handle the interactive menu."""
    detector = train_model()
    
    while True:
        clear_screen()
        print_header()
        print("1. Check single email")
        print("2. Check multiple emails")
        print("3. View sample spam patterns")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            clear_screen()
            print_header()
            print("Enter the email text (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            text = " ".join(lines)
            if text.strip():
                result = detector.predict(text)
                print_result(text, result)
                input("\nPress Enter to continue...")
            
        elif choice == '2':
            clear_screen()
            print_header()
            print("Enter emails (one per line). Type 'DONE' when finished:")
            
            emails = []
            while True:
                email = input()
                if email.upper() == 'DONE':
                    break
                if email.strip():
                    emails.append(email)
            
            if emails:
                print("\nAnalyzing emails...")
                for i, email in enumerate(emails, 1):
                    result = detector.predict(email)
                    print(f"\nEmail #{i}: {email[:100]}{'...' if len(email) > 100 else ''}")
                    print(f"Classification: {'SPAM' if result['is_spam'] else 'NOT SPAM'}")
                    print(f"Spam Probability: {result['spam_probability']:.2%}")
                    print("-" * 30)
                input("\nPress Enter to continue...")
            
        elif choice == '3':
            clear_screen()
            print_header()
            print("Common Spam Patterns to Watch For:")
            patterns = [
                "Excessive use of capital letters",
                "Urgency words (Act now, Limited time)",
                "Money-related promises",
                "Requests for personal information",
                "Suspicious links or attachments"
            ]
            for i, pattern in enumerate(patterns):
                print(f"{i + 1}. {pattern}")
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            clear_screen()
            print_header()
            print("Thank you for using Email Spam Detector!")
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        clear_screen()
        print("\nProgram terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
