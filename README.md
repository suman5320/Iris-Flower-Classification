# Iris-Flower-Classification

This project performs classification of the famous Iris flower dataset using Support Vector Machine (SVM). It includes full **Exploratory Data Analysis (EDA)**, **feature selection**, **model training**, and **hyperparameter tuning using GridSearchCV**.

# Project Structure


---

## 📊 Dataset


- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target**: Species (`setosa`, `versicolor`, `virginica`)

# Key Steps

# Exploratory Data Analysis (EDA)
- Data shape, types, and missing values
- Univariate plots (histogram, boxplots)
- Bivariate plots (pairplot, scatterplot)
- Correlation heatmap and feature relationships
- Class-wise statistical summary

# Feature Engineering
- Dropped `sepal_width` (low correlation & less useful in separation)

# Model: Support Vector Machine (SVC)
- Kernel tried: `linear`, `rbf`, `poly`
- Used `GridSearchCV` for tuning `C`, `gamma`, `kernel`
- Best performance achieved with:
  - `kernel='linear'`
  - `C=1`
  - `gamma='scale'`

# Evaluation
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Accuracy on test set: **94%**

---

# Result & Observations

- The **linear kernel** outperformed RBF and polynomial kernels.
- `C=1` generalized better than `C=10`, even though GridSearchCV selected `C=10` due to slightly higher CV score.
- `sepal_width` was found to be the least useful feature in classification performance.
- Final model was **simple, fast, and accurate**.

---

# Model Saving & Loading

Model is saved using `joblib`:

from joblib import dump, load
dump(model, 'svm_model.joblib')
model = load('svm_model.joblib')
