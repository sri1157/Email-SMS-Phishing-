


# **Phishing Detection using Machine Learning**  

## **Overview**  
This project is a machine learning-based phishing detection system that analyzes emails and URLs to classify them as either *phishing* or *legitimate*. It uses **Logistic Regression** with **text vectorization** techniques to extract features from email content and URLs.  

## **Features**  
- Extracts **domain names** from URLs.  
- Converts email and URL data into **numerical feature vectors** using **CountVectorizer**.  
- Trains a **Logistic Regression model** for classification.  
- Performs **cross-validation** to ensure model reliability.  
- Provides **accuracy and classification reports** for performance evaluation.  
- Allows **user input** to predict whether a given email or URL is phishing.  

## **Requirements**  
Ensure you have the following Python libraries installed:  
```bash
pip install pandas scikit-learn scipy
```  

## **Dataset**  
The program loads phishing data from a CSV file (`pish.csv`). Ensure this file is available in the same directory before running the script.  

## **How It Works**  
1. **Load Dataset** – Reads phishing-related data from `pish.csv`.  
2. **Preprocess Data** – Extracts domain names, combines email and URL text, and converts them into feature vectors.  
3. **Train Model** – Uses **Logistic Regression** with stratified train-test splitting.  
4. **Evaluate Model** – Computes **accuracy** and **classification metrics**.  
5. **Predict Phishing Attempts** – Accepts user input (email/URL) and classifies it as *Phishing* or *Legitimate*.  

## **Usage**  
Run the script using:  
```bash
python new_phishing.py
```  
Follow the on-screen prompts to check whether an email or URL is phishing.  

## **Example Usage**  
```
Do you want to check an email address or a URL?
1. Email Address
2. URL
Enter 1 for Email or 2 for URL: 1
Enter the email address to check: example@fake.com
The email address is likely: Phishing
```

## **Model Performance**  
- Displays **cross-validation accuracy**.  
- Prints a **detailed classification report** including precision, recall, and F1-score.  

## **Future Enhancements**  
- Improve feature engineering using **TF-IDF** instead of CountVectorizer.  
- Train with **deep learning models** (LSTMs, transformers).  
- Integrate with **real-time phishing detection APIs**.  

## **Contributors**  

D.Srivallika

T.Pranay Sai

M.Tharun

E.Prajvin


- Internship Project at HYDERABAD INSTITUTE OF TECHNOLOGY & MANAGEMENT.  



