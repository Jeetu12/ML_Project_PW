# ML_Project_PW

**`README.md`** for GitHub repository:

# Cryptocurrency Liquidity Prediction using Machine Learning

This project utilizes historical cryptocurrency data to **predict liquidity ratios**, enabling traders and exchanges to anticipate potential market instability. It includes data preprocessing, feature engineering, model training, evaluation, and deployment using both **Flask API** and **Streamlit web app**.

## Problem Statement

- Cryptocurrency markets are **highly volatile**, and **liquidity** plays a critical role in price stability. Predicting liquidity helps exchanges manage risks and informs traders during high uncertainty.

# **Data Processing Pipeline:-**

- Data processing pipeline for the liquidity prediction model. Raw market data are ingested (e.g., from CoinGecko APIs), then cleaned (handling missing values, normalizing numeric data) and transformed into features (e.g., technical indicators, volatility, and liquidity ratios). These features feed into model training (e.g., Random Forest, XGBoost) and evaluation stages, after which the selected model is served via an application (Streamlit UI calling a Flask API). This follows the standard ETL/ELT pattern (extract-transform-load) for machine-learning pipelines. In the deployment stage, a front-end (Streamlit) interacts with a back-end (Flask API) that loads the trained model and returns predictions.

---

## **1. Feature Distributions:**

- Histograms of key numeric features (e.g., trading volume, price changes, transaction counts) illustrate each feature’s distribution and reveal outliers. Such plots (histograms or density plots) let us **visualize the range, skewness, and outliers** of each feature.  In practice, our EDA includes one histogram per feature (with kernel density overlays) to ensure no extreme values or unexpected distributions.  For example, a long tail in a volume histogram indicates rare but extreme market activity that may need winsorizing or transformation.

---

## **2. Feature Correlation Heatmap:**

- Correlation heatmap of the engineered features. Each cell’s color shows the pairwise Pearson correlation (dark = strong positive, light = strong negative). This allows us to identify features that move together (e.g., price and volume), which may inform feature selection or multicollinearity checks.  In the heatmap above, we can easily identify clusters of highly correlated inputs (e.g., “Volatility” vs. “Trading Volume” in a real crypto dataset) versus those with near-zero correlations.  In general, a correlation matrix and heatmap help summarize interdependencies among all numeric variables, guiding dimensionality reduction or indicating features to merge or drop.

---

## **3. Feature Importance (Random Forest):**

- Random Forest feature importance scores (mean decrease in accuracy or Gini). Each bar shows how much including that feature improves model accuracy (higher = more influential).  Random Forest feature importance reveals which market factors (e.g., recent volatility, exchange count, transaction frequency) most strongly influence the liquidity prediction.  In our example, a high importance score for “Order Book Spread” or “Volume” means shuffling that feature’s values greatly reduces model accuracy.  This bar chart ranks features by their effect on model performance, confirming which inputs the model relies on most.

---

## **4. Model Comparison (R² and RMSE):**

- To choose the best algorithm, we compare models using metrics like R² (variance explained) and RMSE (prediction error). R² is the proportion of variance in liquidity explained by the model (100% means perfect fit); RMSE is the root mean squared error in original units (smaller is better).  For instance, a Random Forest might achieve R²≈0.85 with RMSE≈0.6, while linear regression gets R²≈0.50 and RMSE≈1.0.  We plot each model’s R² and RMSE (bar charts or tables) side by side.  This visualization quickly highlights that models with higher R² and lower RMSE perform best.  (We note R² tends to increase with more features, so we also track adjusted R² when comparing model complexity.)

---

## **5. Evaluation Plots (Residuals and Predictions):**

- Residual plot for the final model. We plot each prediction’s residual (error = actual–predicted) against the expected value.  An ideal residual plot appears as a random cloud centered at zero, indicating no discernible pattern or heteroskedasticity.  In practice, we check that the residuals cluster around zero with roughly equal spread across all predicted values.  Any funnel shape or trend would signal model misspecification.

- Predicted vs. Actual scatter (two example models). Perfect predictions lie on the diagonal line; points tight around that line indicate high accuracy.  In the left panel (R²=0.81), the green points fall near the diagonal, showing a good fit.  In the right panel (R² = 0.24), points deviate widely from the line, indicating a poor model fit.  This chart is a rich diagnostic: a strong linear pattern indicates an accurate model, while deviations or outliers highlight remaining errors.

---

## **6. Deployment Architecture:**

- Our deployed system uses **Streamlit** for the user interface and **Flask** as the model API. In a typical setup, the Streamlit app (web frontend) collects user inputs (e.g., current market features) and calls a Flask REST API endpoint. The Flask server loads the trained model (e.g., from a pickle) and returns the predicted liquidity score as JSON.  Streamlit then displays the result to the user.  Thus, Streamlit handles UI/UX (client), while Flask handles backend serving (model inference).  This separation (frontend vs. backend) ensures a responsive interface: users interact through the Streamlit client, while the Python-based server (Flask) computes predictions in real-time.

---

## Project Structure

CryptoCurrency_Liquidity_Prediction/
│
├── .CSV data files/                        # Raw CSV files from CoinGecko
│
├── Notebooks/
│   └── Crypto\_Liquidity.ipynb   # Google Colab notebook (EDA + modeling)
│
├── Apps/
│   ├── flask\_app.py             # Flask API for deployment
│   └── streamlit\_app.py         # Streamlit web app
│
├── Models/
│   └── crypto\_liquidity\_model.pkl  # Trained ML model
│
├── Reports/
│   ├── EDA\_Report.pdf           # Summary of EDA & insights
│   ├── HLD\_Document.pdf         # High-Level Design
│   └── LLD\_Document.pdf         # Low-Level Design
│
├── README.md
└── requirements.txt

---

# Requirements

# Core ML & Data
numpy
pandas
scikit-learn
matplotlib
seaborn

# For saving/loading models
pickle5  # optional if using older Python

# API & App options
flask     # for Flask API
streamlit # for Streamlit UI

# Optional (PDFs, Visual Reports)
fpdf

# Version pinning (optional for stability)
# numpy==1.24.3
# scikit-learn==1.3.0
# flask==2.3.3
# streamlit==1.35.0


---

## Features & Target

- **Input Features**:

  - `price`, `1h`, `24h`, `7d` % changes

  - `24h_volume`, `market_cap`

  - `volatility` (engineered as `|24h|`)

- **Target**:

  - `liquidity_ratio = volume / market_cap`

---

## Model Used

- `RandomForestRegressor`

- `GridSearchCV` for hyperparameter tuning

- Metrics:

  - **R² Score**: ~0.94

  - **RMSE**: Low, consistent performance

## Deployment

### Streamlit App

https://mlprojectpw-pu4pcs5gfjlqdcmv2ef5ld.streamlit.app/

### Flask API

https://ml-project-pw.onrender.com/

POST Request Example:

```json
{
  "price": 40859.46,
  "1h": 0.022,
  "24h": 0.03,
  "7d": 0.055,
  "24h_volume": 35000000000,
  "mkt_cap": 770000000000
}
```
## Sample Output

Predicted Liquidity Ratio: 0.045832

## Future Enhancements

* Add an LSTM or XGBoost model for improved accuracy

* Integrate live data from APIs

* Extend time series trends using moving averages

## Tech Stack

* Python, Pandas, Scikit-learn

* Google Colab

* Streamlit, Flask

* Matplotlib, Seaborn

## Author

**Arijit Chakraborty** — [EduFinTech](https://www.edufintech.co.in)


