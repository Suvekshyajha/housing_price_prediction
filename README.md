Housing Price Analysis & ML Pipeline
EDA and ML pipeline on Housing.csv — univariate/bivariate/multivariate analysis, label encoding, correlation heatmap, VIF check, sklearn preprocessing pipeline, and logistic regression model with train/test split and classification report.

📁 Project Structure
├── sample.ipynb       # Main analysis notebook
├── Housing.csv        # Dataset (place in same directory)
└── README.md

📊 What This Notebook Does
1. Data Loading & Exploration

Loads Housing.csv with pandas(exported from kaggle)
Separates numerical and categorical columns
Descriptive stats on price and area

2. Univariate Analysis

Histogram distributions for all numerical columns (with skewness)
Count plots for all categorical columns

3. Bivariate Analysis (vs Price)

area vs price — regression plot (positive correlation, right-skewed)
stories vs price — box plot
bedrooms vs price — box plot
bathrooms vs price — box plot
parking vs price — box plot
All categorical features vs price — box plots

4. Data Cleaning

Missing value check (count + percentage per column)

5. Encoding

Binary columns (yes/no) → mapped to 1/0
furnishingstatus → ordinal encoded: unfurnished=0, semi-furnished=1, furnished=2

6. Multivariate Analysis

Correlation heatmap (all numeric features, YlGnBu palette)
VIF (Variance Inflation Factor) to check multicollinearity

7. Preprocessing Pipeline (sklearn)

SimpleImputer (median strategy) + StandardScaler for numeric features
ColumnTransformer combining numeric and categorical pipelines

8. Model Training

train_test_split (80/20, random_state=42)
LogisticRegression on preprocessed data
Evaluation: Accuracy score + Classification report


⚙️ Requirements
bashpip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy

🚀 How to Run

Place Housing.csv in the same folder as sample.ipynb
Install requirements above
Open and run all cells in order:

bashjupyter notebook sample.ipynb

📌 Key Findings

Area has the strongest positive correlation with price
Bathrooms show a steep positive price trend
Stories and bedrooms also positively correlate with price
Dataset has no missing values
Right-skewed distributions in most numeric features


⚠️ Notes

The notebook uses LogisticRegression which is suited for classification — if price is continuous, consider switching to LinearRegression or Ridge
