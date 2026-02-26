# 🎬 IMDb Movies Data Analysis & Prediction

## 📌 Project Overview

This project was developed during my second year (second semester) as a Data Science student — approximately one year and two months ago.  
It reflects my early hands-on experience applying data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning techniques to a real-world dataset.

The dataset contains information about movies and series including:
- Title information
- Runtime
- Genres
- Release year
- Ratings
- Number of votes

---

# 1️⃣ What Was the Problem?

The main objective of this project was:

- To analyze movie data and extract meaningful insights.
- To understand how runtime, genre, and release year relate to ratings and popularity.
- To build a predictive model to estimate the number of votes a movie might receive based on selected features.

In short:
> Can we predict a movie’s popularity (number of votes) using runtime and rating-related features?

---

# 2️⃣ Data Challenges

While working with the dataset, several issues appeared:

### 🔹 Missing Values
- `startYear` had invalid values converted to null.
- `runtimeMinutes` contained many missing entries.
- Some categorical values were represented as `\N`.

### 🔹 Outliers
- Extreme runtime values (some exceeding 5000 minutes).
- Highly skewed distribution in `numVotes` (very large vote counts for some movies).

### 🔹 Data Type Inconsistencies
- Some numeric columns were stored as strings and needed conversion.

---

# 3️⃣ My Decisions

To handle these issues, I made the following decisions:

### ✅ Data Cleaning
- Converted numeric columns using `pd.to_numeric()`.
- Dropped rows with critical missing values in `startYear`.
- Replaced missing runtime values with the median (robust against outliers).
- Replaced missing genres with the most frequent genre.

### ✅ Outlier Handling
- Used IQR (Interquartile Range) to detect extreme runtime values.
- Analyzed distributions using boxplots and histograms before deciding how to handle them.

### ✅ Feature Engineering
- Extracted `primaryGenre`.
- Created `runtimeCategory` (Short / Medium / Long).
- Added interaction feature: `runtime_x_rating`.
- Created a binary column `isSuccessful` (rating ≥ 6).

### ✅ Exploratory Data Analysis
- Distribution of movies over years.
- Relationship between runtime and rating.
- Votes vs rating (log scale).
- Success rate by runtime category.
- Top genres by rating and popularity.

---

# 4️⃣ Why I Chose These Models

I experimented with different regression models to predict `numVotes`:

### 🔹 Linear Regression
Used as a baseline model to understand linear relationships.

### 🔹 Random Forest Regressor
Chosen because:
- It handles non-linearity well.
- It is robust to outliers.
- It captures feature interactions automatically.

### 🔹 XGBoost Regressor
Selected for:
- Strong performance in structured/tabular data.
- Ability to handle complex relationships.

Model performance was evaluated using RMSE (Root Mean Squared Error).

---

# 📊 Key Insights

- Medium-length movies showed slightly higher average ratings.
- Votes distribution is highly skewed.
- Rating and number of votes have a visible relationship when plotted on a log scale.
- Runtime alone is not sufficient to strongly predict popularity.

---

# 🧠 What This Project Represents

This project represents:
- My early practical experience in Data Science.
- My understanding of data cleaning and EDA.
- My first experimentation with multiple machine learning models.
- My ability to reason about data problems and make decisions.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

# 🚀 Future Improvements

If revisited, I would:

- Apply log transformation to `numVotes`.
- Perform cross-validation.
- Add more feature engineering.
- Try classification for predicting movie success.
- Perform deeper model evaluation (R², MAE, feature importance).
