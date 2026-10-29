# SMS Spam Classifier using Naive Bayes

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Multinomial Naive Bayes and text feature extraction with CountVectorizer.



## Project Overview

Spam messages often contain promotional offers, suspicious links, or unwanted advertisements. This project uses Natural Language Processing (NLP) techniques and a supervised Machine Learning algorithm to identify potential spam messages based on patterns learned from labeled SMS data.
The project covers the complete ML workflow, from data exploration and text vectorization to model training, evaluation, and hyperparameter tuning.



## Objectives

- Understand and preprocess a real-world SMS dataset.
- Convert text messages into numerical features using CountVectorizer.
- Train a Multinomial Naive Bayes classifier.
- Evaluate classification performance using multiple metrics.
- Optimize the smoothing parameter using GridSearchCV.
- Save the trained model and vectorizer for future predictions.



## Dataset

**Dataset:** SMS Spam Collection Dataset
**Source:** [Kaggle — SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset contains 5,572 SMS messages labeled as:
- **Ham:** Legitimate messages.
- **Spam:** Unwanted or suspicious messages.



### Dataset Distribution

| Class | Number of Messages | Percentage |
|---|---:|---:|
| Ham | 4,825 | 86.59% |
| Spam | 747 | 13.41% |
| **Total** | **5,572** | **100%** |

The dataset is imbalanced, with legitimate messages making up the majority of observations. Therefore, accuracy is evaluated alongside precision, recall, and F1-score.



## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook
- Natural Language Processing (NLP)
- Machine Learning



## Project Workflow


### 1. Data Loading and Exploration
- Loaded the dataset using Pandas.
- Inspected dataset dimensions, column types, and missing values.
- Retained the relevant message and label columns: `v2` and `v1`.
- Examined the distribution of ham and spam messages.

### 2. Train-Test Split
Split the dataset into training and testing subsets.

- Training data: 80%
- Testing data: 20%
- `random_state=42` for reproducibility.
- `stratify=y` to preserve the approximate class distribution in both subsets.

The resulting datasets contained 4,457 training messages and 1,115 testing messages.



### 3. Text Feature Extraction
Used `CountVectorizer` to convert SMS text into numerical word-count features.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
```

The vectorizer learned a vocabulary of **7,701 features** from the training messages.

`fit_transform()` learns the vocabulary and transforms the training data. `transform()` applies the same vocabulary to the test data, preventing vocabulary leakage from the test set.


### 4. Model Training
Trained a Multinomial Naive Bayes classifier, which is well suited to text classification tasks involving word-count features.

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train_counts, y_train)
```

The initial model used the default smoothing parameter, `alpha=1.0`.


### 5. Model Evaluation
Evaluated predictions using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

These metrics provide a more complete assessment than accuracy alone, particularly because spam messages are the minority class.


### 6. Hyperparameter Tuning
Compared multiple values of `alpha`, which controls Laplace smoothing in Multinomial Naive Bayes.

Used `GridSearchCV` with five-fold cross-validation and macro F1-score to select the hyperparameter.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

param_grid = {
    "alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
}

grid_search = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_search.fit(X_train_counts, y_train)

best_nb_model = grid_search.best_estimator_
```

**Best parameter found:** `alpha = 0.01`

**Best cross-validation macro F1-score:** approximately `96.30%`.

The selected estimator was evaluated on the held-out test set.


## Final Model Performance
The tuned model achieved the following results on the test set.

| Metric | Score |
|---|---:|
| Accuracy | 98.48% |
| Spam Precision | 96.48% |
| Spam Recall | 91.95% |
| Spam F1-score | 94.15% |
| Macro F1-score | Approximately 96.5% |

### Confusion Matrix

Rows represent actual labels and columns represent predicted labels, in the order `ham`, `spam`.

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 961 | 5 |
| Spam | 12 | 137 |

### Results Interpretation

- Correctly classified 1,098 out of 1,115 test messages.
- Correctly identified 137 of the 149 actual spam messages.
- Missed 12 spam messages.
- Incorrectly flagged 5 legitimate messages as spam.

The results indicate strong performance on this dataset. However, real-world performance may vary with new message styles, unseen vocabulary, and changes in spam patterns.

## Custom Message Prediction

The trained model was also tested on custom SMS examples.

| Example Message | Prediction |
|---|---|
| Congratulations! You won a free prize. Call now! | Spam |
| Hey bro, are you coming to college tomorrow? | Ham |
| URGENT! Claim your free cash reward now | Spam |
| Hi, please call me when you reach home | Ham |

These examples demonstrate how the trained model can classify new messages. They are illustrative checks, not a substitute for broader testing.

## Saving the Model

Saved both the trained classifier and the fitted vectorizer using Joblib.

```python
import joblib

joblib.dump(best_nb_model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "count_vectorizer.pkl")
```

The vectorizer must be saved alongside the model because new SMS messages need to be transformed using the same learned vocabulary and feature ordering.

## Key Learnings

Through this project, I learned:

- How to prepare and explore a real-world classification dataset.
- Why train-test splitting and stratification matter.
- How text is transformed into numerical features.
- How Multinomial Naive Bayes uses word-count information.
- The difference between accuracy, precision, recall, and F1-score.
- How confusion matrices reveal false positives and false negatives.
- How Laplace smoothing works and how `alpha` affects predictions.
- How to tune model hyperparameters with GridSearchCV and cross-validation.
- How to save a trained model for reuse.

## Limitations

- The model relies on word-count features and does not fully understand message meaning or context.
- Unseen words are not represented by the fitted vocabulary.
- Performance may change as real-world spam patterns evolve.
- Evaluation was performed on a single held-out split from this dataset.
- The model is a learning project and should not be treated as a complete production-grade spam-filtering system.

## Future Improvements

- Build an interactive web interface using Streamlit or FastAPI.
- Compare CountVectorizer with TF-IDF features.
- Compare Multinomial Naive Bayes with Logistic Regression and other text classifiers.
- Evaluate different decision thresholds and their effects on precision and recall.
- Test the model on additional, more recent SMS data.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for SMS spam classification, including text feature extraction, model training, evaluation, hyperparameter tuning, and model persistence.

The tuned Multinomial Naive Bayes classifier achieved **98.48% test accuracy** and a **94.15% spam F1-score** on the selected test split.

The project strengthened my understanding of text classification, probabilistic learning, model evaluation, and hyperparameter optimization.

---

**Author:** Nagaraj

**Domain:** Machine Learning | Natural Language Processing | Text Classification
