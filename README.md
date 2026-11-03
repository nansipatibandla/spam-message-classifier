# Spam Message Classifier

A machine learning model that classifies SMS messages as spam or not spam, built for the CodeNova AI/ML Internship (Task 2).

## Features
- Text preprocessing with TF-IDF (lowercasing, tokenization, stop-word removal)
- Compares Multinomial Naive Bayes and Logistic Regression
- Evaluation on a held-out test set (80/20 split, no data leakage)
- Saved model and vectorizer (joblib)
- Interactive function for testing custom messages

## Tech Stack
Python, scikit-learn, pandas, matplotlib, seaborn, joblib

## Dataset
SMS Spam Collection (UCI): 5,572 messages (4,825 ham, 747 spam).

## Setup
```
pip install scikit-learn pandas matplotlib seaborn nltk joblib ipykernel
```
Open `spam_classifier.ipynb` and run all cells.

## Results (test set, 1,115 messages)

| Model | Spam precision | Spam recall | Spam F1 | Accuracy |
|---|---|---|---|---|
| Naive Bayes (final) | 1.00 | 0.78 | 0.88 | 0.97 |
| Logistic Regression | 1.00 | 0.76 | 0.86 | 0.97 |

Both models reach precision 1.00 on spam, so no legitimate message was marked as spam.

## Confusion Matrices
![Naive Bayes](cm_Naive_Bayes.png)
![Logistic Regression](cm_Logistic_Regression.png)

## Sample Predictions
```
NOT SPAM (spam probability: 0.46) <- URGENT! Your account is suspended, call 0906123456
SPAM (spam probability: 0.98) <- WINNER!! As a valued network customer you have bee
SPAM (spam probability: 0.83) <- Free entry in 2 a weekly comp to win FA Cup ticket
NOT SPAM (spam probability: 0.00) <- Hey, are we meeting at 5 today?
NOT SPAM (spam probability: 0.04) <- Can you send me the notes from class?
```

## Interactive Demo
```
Type a message to check (or type 'quit' to stop)
-> NOT SPAM (spam probability: 0.46)
-> SPAM (spam probability: 0.80)
-> NOT SPAM (spam probability: 0.04)
-> SPAM (spam probability: 0.76)
Done.
```

## Limitations
- Recall is about 0.78, so roughly 1 in 5 spam messages is missed (for example, short account-suspension style messages scored near the cutoff).
- The model learned only from this dataset, so spam in a different style may be missed.
