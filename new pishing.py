import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import re
from scipy.sparse import hstack

# Load the CSV from local directory
df = pd.read_csv('pish.csv')  # Update 'pish.csv' to the correct path of your local file

# Check for missing values (if any)
print(df.isnull().sum())

# Extract domain from URL
def extract_domain(url):
    domain = re.findall(r'(?<=://)([A-Za-z0-9.-]+)(?=\/|$)', url)
    return domain[0] if domain else ''

# Concatenate email and URL into one text feature and add domain as a feature
df['combined'] = df['email'] + ' ' + df['url']
df['domain'] = df['url'].apply(extract_domain)

# Split data into features (X) and labels (y)
X = df[['combined', 'domain']]
y = df['label']

# Feature extraction: Convert text data into numerical feature vectors
vectorizer_combined = CountVectorizer()
X_combined_vectorized = vectorizer_combined.fit_transform(X['combined'])

vectorizer_domain = CountVectorizer()
X_domain_vectorized = vectorizer_domain.fit_transform(X['domain'])

# Combine vectorized features
X_vectorized = hstack([X_combined_vectorized, X_domain_vectorized])

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# Train the Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Cross-validation
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
cv_scores = cross_val_score(model, X_vectorized, y, cv=sss)
print(f"Cross-validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Show the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict whether a user-inputted email or URL is phishing or legitimate
def predict_phishing_email(email):
    """Predict if the provided email is phishing or legitimate."""
    combined_input = email + ' ' + ''  # Only email input
    combined_input_vectorized = vectorizer_combined.transform([combined_input])
    domain_input_vectorized = vectorizer_domain.transform([''])  # Dummy domain input for consistency
    input_vectorized = hstack([combined_input_vectorized, domain_input_vectorized])
    prediction = model.predict(input_vectorized)
    result = "Phishing" if prediction[0] == 0 else "Legitimate"
    return result

def predict_phishing_url(url):
    """Predict if the provided URL is phishing or legitimate."""
    domain = extract_domain(url)
    combined_input = '' + ' ' + domain  # Only domain input
    combined_input_vectorized = vectorizer_combined.transform([combined_input])
    domain_input_vectorized = vectorizer_domain.transform([domain])
    input_vectorized = hstack([combined_input_vectorized, domain_input_vectorized])
    prediction = model.predict(input_vectorized)
    result = "Phishing" if prediction[0] == 0 else "Legitimate"
    return result

# Check email or URL using predefined inputs
def main():
    # User interaction for choosing to check email or URL
    print("Do you want to check an email address or a URL?")
    print("1. Email Address")
    print("2. URL")

    choice = input("Enter 1 for Email or 2 for URL: ")

    if choice == '1':
        email_input = input("Enter the email address to check: ")
        result = predict_phishing_email(email_input)
        print(f"\nThe email address is likely: {result}")

    elif choice == '2':
        url_input = input("Enter the URL to check: ")
        result = predict_phishing_url(url_input)
        print(f"\nThe URL is likely: {result}")

    else:
        print("Invalid choice. Please enter 1 for Email or 2 for URL.")

    # Displaying accuracy and classification report again
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Call the main function
if __name__ == '__main__':
    main()
