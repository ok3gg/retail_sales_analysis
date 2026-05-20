# retail_sales_analysis
Retail sales forecasting and trend analysis project using Python, scikit-learn, and Streamlit.

## Project Overview
This project analyzes historical retail sales data from the Rossmann Store Sales dataset to identify sales trends, seasonal demand patterns, promotional effects, and customer behavior relationships.

The project combines exploratory data analysis, regression analysis, machine learning models, and time-series forecasting techniques to support retail sales prediction and operational planning.


## Project Objectives

- Analyze retail sales trends and seasonality
- Evaluate the impact of promotions and holidays on sales
- Develop predictive machine learning models for sales forecasting
- Compare regression and ensemble learning approaches
- Build an interactive Streamlit forecasting dashboard
- Generate business insights for operational planning


## Dataset

Dataset used:
Rossmann Store Sales Dataset (Kaggle)
https://www.kaggle.com/competitions/rossmann-store-sales/data

The dataset contains:
- daily retail sales
- customer counts
- promotions
- holidays
- store operational information
- time-series sales records


## Analysis Performed

### Exploratory Data Analysis
- Daily and monthly sales trends
- Promotion impact analysis
- Holiday impact analysis
- Correlation analysis
- Sales distribution analysis

### Predictive Modeling
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

### Forecasting Dashboard
- Holt-Winters Exponential Smoothing
- Interactive store-level forecasting
- Forecast horizon controls
- Interactive Plotly visualizations


## Key Findings

- Customer traffic was identified as the strongest predictor of retail sales.
- Promotional campaigns demonstrated a significant positive relationship with sales performance.
- Random Forest Regression achieved the strongest predictive performance among the evaluated models.
- Retail sales exhibited recurring seasonal and weekly demand patterns.


## Interactive Forecasting Dashboard

An interactive Streamlit dashboard was developed to provide store-level retail sales forecasting and visualization capabilities. (app.py)

Dashboard features include:
- store selection filtering
- adjustable forecast horizons
- historical sales visualization
- future sales forecasting
- operational KPI metrics
- interactive Plotly charts


## Dashboard Preview


## Project Structure

```
project/
│
├── src/
│   ├── retail_sales_analysis.ipynb
│   └── app.py
│
├── data/
│
├── images/
│   └── dashboard.png
│
├── README.md
│
└── requirements.txt
```
