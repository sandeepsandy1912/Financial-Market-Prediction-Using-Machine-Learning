### **📌 Financial Market Prediction Using Machine Learning**  


## 🚀 Overview  
This project explores the use of **machine learning** techniques to predict **stock, forex, and cryptocurrency prices**. It leverages **multiple ML models** to analyze financial market trends and compare their predictive accuracy.

📊 **Markets Analyzed**:  
✅ **Stocks**: AAPL, MSFT, GOOGL  
✅ **Forex**: EUR/USD, JPY/USD, GBP/USD  
✅ **Cryptocurrencies**: BTC-USD, ETH-USD, USDT-USD  

🔬 **Models Used**:  
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Regression (SVR)**
- **Ridge Regression**
- **Long Short-Term Memory (LSTM)** (Neural Network)

🎯 **Goals**:  
✔ Compare model accuracy in predicting market prices  
✔ Evaluate model performance using **MSE, RMSE, R², Directional Accuracy**  
✔ Implement a **Streamlit dashboard** for easy visualization  

---

## 📂 Project Structure  
```bash
financial-market-prediction/
│── data/                           # Data files
│── models/                         # Saved trained models 
│── notebooks/                      # Jupyter Notebooks for EDA & analysis
│── src/                            # Python scripts for model training & evaluation
│    ├── data_preprocessing.py      # Data cleaning & feature engineering
│    ├── model_training.py          # Machine learning models
│    ├── evaluation.py              # Model evaluation & metrics
│    ├── streamlit_dashboard.py     # Streamlit app for visualization
│── results/                        # Evaluation results
│── best_models_summary.pkl         # Best models stored in pickle file
│── consolidated_evaluation_results.pkl  # Consolidated evaluation results
```

---

## 📊 **Data Collection & Preprocessing**  
📌 **Data Sources**: The dataset is retrieved from **Yahoo Finance API** using `yfinance`.  
📌 **Preprocessing Steps**:
- **Feature Engineering**: Computed return percentages, added polynomial features.
- **Normalization**: Standardized prices, volume, and return data.
- **Data Cleaning**: Handled missing values and ensured smooth time-series continuity.
- **Splitting**: Used **80-20 train-test split** to evaluate model performance.

---

## 🔥 **Machine Learning Models Used**  

| Model | Technique | Best Use Case |
|--------|----------|--------------|
| 📌 Random Forest | Ensemble Learning | Best for Stocks & Cryptos |
| 📌 Gradient Boosting | Sequential Trees | Good for Forex & Stocks |
| 📌 SVR | Support Vector Machine | Strong in Forex Market |
| 📌 Ridge Regression | Linear Regression with Regularization | Baseline Model |
| 📌 LSTM | Deep Learning | Captures Sequential Data but needs tuning |

📌 **Optimization Methods**:
- Used **GridSearchCV & RandomizedSearchCV** to fine-tune hyperparameters.
- **LSTM optimized using Keras Tuner**.

---

## 📏 **Model Evaluation Metrics**  

Each model was evaluated using **multiple statistical measures**:

| Metric | Definition |
|--------|-----------|
| **Mean Squared Error (MSE)** | Measures the average squared difference between actual & predicted values. |
| **Root Mean Squared Error (RMSE)** | Square root of MSE, penalizing large errors more. |
| **R² Score** | Measures how well model predictions match actual values. |
| **Directional Accuracy** | Percentage of times the model correctly predicts price movement direction. |

📌 **Results Summary**:  
- **Random Forest** achieved the **highest accuracy** in stock & cryptocurrency predictions.  
- **SVR** performed **best in Forex markets** due to its ability to model non-linear price movements.  
- **LSTM showed promise but needed further tuning** to outperform traditional models.  

---

## 🖥 **Streamlit Dashboard** 🎨  

A **Streamlit application** was developed to visualize **actual vs. predicted price movements**.  

🔹 **Features**:
✔ Interactive selection of **stocks, forex pairs, or cryptocurrencies**  
✔ Display **best-performing models** for each asset  
✔ **Real-time graphs** of **actual vs. predicted prices**  
✔ **Downloadable plots** for further analysis  

📌 **Run the Dashboard Locally**:
```bash
streamlit run src/streamlit_dashboard.py
```


## 🚀 **Results Summary**
📊 **Best Models for Each Market**:

| Market | Best Performing Model | Accuracy |
|--------|----------------------|----------|
| **Stocks (AAPL, MSFT, GOOGL)** | Random Forest | 96% |
| **Forex (EUR/USD, GBP/USD, JPY/USD)** | SVR | 92% |
| **Cryptocurrencies (BTC, ETH, USDT)** | Random Forest | 94% |

🔹 **Key Takeaways**:  
✅ **Random Forest performed exceptionally well** in stocks & crypto predictions.  
✅ **SVR was the best for Forex market predictions**.  
✅ **LSTM requires further optimization for financial markets**.  

---

## 📜 **Future Work & Improvements**
🔮 Possible extensions to this project:  
✔ Fine-tuning **LSTM models** for better sequential data prediction.  
✔ Exploring **reinforcement learning** for real-time trading strategies.  
✔ Adding **sentiment analysis** on news headlines to improve accuracy.  
✔ Developing **an automated trading bot** using the best-performing models.  

---
