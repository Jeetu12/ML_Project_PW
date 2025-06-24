# ML_Project_PW

**`README.md`** for GitHub repository:


```markdown

# Cryptocurrency Liquidity Prediction using Machine Learning

- This project uses historical cryptocurrency data to **predict liquidity ratios**, helping traders and exchanges anticipate potential market instability. It includes data preprocessing, feature engineering, model training, evaluation, and deployment using both **Flask API** and **Streamlit web app**.

---

## Problem Statement

- Cryptocurrency markets are **highly volatile**, and **liquidity** plays a critical role in price stability. Predicting liquidity helps exchanges manage risks and informs traders during times of high uncertainty.

---

## Project Structure

```

crypto-liquidity-prediction/
│
├── data/                        # Raw CSV files from CoinGecko
│
├── notebooks/
│   └── Crypto\_Liquidity.ipynb   # Google Colab notebook (EDA + modeling)
│
├── app/
│   ├── flask\_app.py             # Flask API for deployment
│   └── streamlit\_app.py         # Streamlit web app
│
├── models/
│   └── crypto\_liquidity\_model.pkl  # Trained ML model
│
├── reports/
│   ├── EDA\_Report.pdf           # Summary of EDA & insights
│   ├── HLD\_Document.pdf         # High-Level Design
│   └── LLD\_Document.pdf         # Low-Level Design
│
├── README.md
└── requirements.txt

````

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

- ✅ `RandomForestRegressor`

- 🔁 `GridSearchCV` for hyperparameter tuning

- 📈 Metrics:

  - **R² Score**: ~0.94

  - **RMSE**: Low, consistent performance

---

## Deployment

### Streamlit App

```bash
streamlit run app/streamlit_app.py
````

### ✅ Flask API

```bash
python app/flask_app.py
```

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

---

## Sample Output

```
Predicted Liquidity Ratio: 0.045832
```

---

## Future Enhancements

* Add LSTM or XGBoost model for improved accuracy

* Integrate live data from APIs

* Extend time series trends using moving averages

---

## Tech Stack

* Python, Pandas, Scikit-learn

* Google Colab

* Streamlit, Flask

* Matplotlib, Seaborn

---

## Author

**Arijit Chakraborty** — [EduFinTech](https://www.edufintech.co.in)

---

## License

MIT License

---
