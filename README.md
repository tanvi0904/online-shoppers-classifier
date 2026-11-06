# Online Shopper Purchase Prediction — ML Assignment 2

**Live app:** https://online-shoppers-classifiergit-dp2hbdrk2zqedzknuxvu78.streamlit.app/

## a. Problem statement

Given the browsing behaviour of a single session on an e-commerce website, predict whether that session ends in a purchase (`Revenue = True`) or not. It's a **binary classification** problem. Only about 15% of sessions convert, so the classes are imbalanced, which is why I look at AUC, F1 and MCC as well as accuracy.

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data and compared, and they're served through a Streamlit app that scores an uploaded test CSV using the saved models.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (Sakar et al., 2018)
- **Size:** 12,330 sessions × 17 features + 1 target (each session belongs to a different user over a 1-year period). 125 duplicate rows are dropped, leaving 12,205.
- **Target:** `Revenue` (True / False) — ~84.5% False, ~15.5% True

| Feature group | Columns | Notes |
|---|---|---|
| Page counts & time spent | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | Number of pages of each type visited and total seconds spent; heavily right-skewed |
| Google Analytics metrics | `BounceRates`, `ExitRates`, `PageValues` | Average bounce/exit rate of pages visited; `PageValues` = avg value of pages visited before a transaction |
| Calendar | `SpecialDay`, `Month`, `Weekend` | `SpecialDay` = closeness to a holiday (0–1) |
| Visitor / tech | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | Integer codes are treated as categories; `VisitorType` = Returning / New / Other |

**Preprocessing** (inside one sklearn `Pipeline`, saved with each model):
- Dropped duplicate rows
- `log1p` + standard scaling on the skewed count/duration/`PageValues` columns
- Standard scaling on the rate columns
- One-hot encoding for the categorical columns (`handle_unknown="ignore"`), used by every model except Naive Bayes (see below)
- Stratified 80/20 train/test split (`random_state=42`); the 20% test split (2,441 rows, 382 purchases) is saved as `test_data.csv`

## c. GitHub repository

**Repo:** https://github.com/tanvi0904/online-shoppers-classifier

```
online-shoppers-classifier/
├── app.py                  # Streamlit app
├── train_models.py         # trains + evaluates + saves all 5 models
├── ML_Assignment2_Training.ipynb  # notebook that runs training + checks
├── shopper_utils.py        # shared feature lists, cleaning, metrics
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test split (features + Revenue)
├── data/online_shoppers_intention.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv         # test-set metrics for all models
    └── metrics.md
```

**Run locally**
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_models.py      # recreates model/*.joblib and test_data.csv
streamlit run app.py
```

The saved models were trained with Python 3.13, scikit-learn 1.6.1 and numpy 2.4.6 (pinned in `requirements.txt`); the Streamlit Cloud app runs on Python 3.13 to match.

## d. Models used and comparison

| Model | Setup |
|---|---|
| Logistic Regression | L2, `class_weight="balanced"` |
| Decision Tree | `class_weight="balanced"`, `max_depth` / `min_samples_leaf` tuned with 5-fold CV on F1 (best: depth 4, min leaf 10) |
| KNN | `n_neighbors` / `weights` tuned with 5-fold CV on F1 (best: k=21, distance weighting) |
| Naive Bayes | Gaussian NB on the numeric columns only, `var_smoothing` tuned with 5-fold CV on F1 (best: 1.0) |
| Support Vector Machine | RBF kernel, C=1, `class_weight="balanced"`, probability outputs for AUC |

**Test-set results** (20% hold-out, positive class = purchase):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8710 | 0.9255 | 0.5601 | 0.8168 | 0.6645 | 0.6041 |
| Decision Tree | 0.8480 | 0.9242 | 0.5085 | 0.8586 | 0.6388 | 0.5807 |
| KNN | 0.8964 | 0.9133 | 0.7186 | 0.5550 | 0.6263 | 0.5737 |
| Naive Bayes | 0.8562 | 0.8651 | 0.5306 | 0.7042 | 0.6052 | 0.5272 |
| Support Vector Machine | 0.8595 | 0.9211 | 0.5319 | 0.8508 | 0.6546 | 0.5971 |

**A note on Naive Bayes.** My first run gave Naive Bayes an accuracy of only 0.278 (MCC 0.138), worse than always predicting "no purchase". The cause was feeding Gaussian NB the one-hot 0/1 columns, which aren't bell-shaped, so it predicted "purchase" for almost every session. I fixed it by training Naive Bayes on the numeric columns only and tuning `var_smoothing`; the table above is the result after that fix. The earlier numbers are in the commit history.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best AUC (0.9255), F1 (0.6645) and MCC (0.6041) of the five. Class balancing gives it a good recall (0.82), but precision is only 0.56, so many predicted buyers don't actually buy. Fast to train, and a strong baseline here. |
| Decision Tree | Depth-4 tree after tuning, so it's simple and easy to explain. Highest recall of the five (0.859) but lowest precision (0.509), so it catches most buyers at the cost of many false alarms. MCC 0.581. |
| KNN | Highest accuracy (0.896) and precision (0.719), but recall is the lowest of all (0.555): it misses almost half of the real purchases. With no class weighting it leans towards predicting "no purchase", which flatters accuracy on imbalanced data. |
| Naive Bayes | Weakest model overall: lowest AUC (0.865) and MCC (0.527). Its independence assumption doesn't hold well, since `BounceRates` and `ExitRates` are strongly related. It also needed the numeric-only fix to be usable at all (accuracy 0.278 before). |
| Support Vector Machine | Very close to Logistic Regression (recall 0.851, F1 0.655, MCC 0.597, AUC 0.921) but the slowest to train because of the probability outputs. The extra complexity doesn't buy a better result here. |
| **Overall winner** | **Logistic Regression**: best AUC, F1 and MCC with the simplest model and fastest training. The gaps are small, especially to SVM (MCC 0.604 vs 0.597), and these come from a single 80/20 split, so I wouldn't read much into differences in the second decimal place. |

## Streamlit app features

- Upload a test CSV (features + `Revenue`); a sample `test_data.csv` can be downloaded from the sidebar
- Dropdown to choose any of the 5 saved models
- Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model
- Confusion matrix heatmap + classification report
- "Compare all models" tab showing every model's metrics side by side on the uploaded data
- Download predictions with purchase probability

Models are only **loaded**; nothing is retrained on the uploaded data.
