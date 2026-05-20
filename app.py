import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide"
)


# PAGE TITLE

st.title("Retail Sales Forecasting Dashboard")

st.write(
    """
    Interactive store-level retail sales forecasting dashboard
    using Holt-Winters Exponential Smoothing.
    """
)



# LOAD DATA

@st.cache_data
def load_data():

    train = pd.read_csv("../data/train.csv")

    train["Date"] = pd.to_datetime(train["Date"])

    train = train[train["Open"] == 1]
    train = train[train["Sales"] > 0]

    return train


train = load_data()



# SIDEBAR CONTROLS

st.sidebar.header("Forecast Settings")

store_list = sorted(train["Store"].unique())

selected_store = st.sidebar.selectbox(
    "Select Store",
    store_list
)

forecast_days = st.sidebar.number_input(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=365,
    value=30,
    step=1
)

st.sidebar.header("Visualization Settings")

history_days = st.sidebar.number_input(
    "Historical Days to Display",
    min_value=30,
    max_value=2000,
    value=180,
    step=30
)


# FILTER STORE DATA

store_data = train[
    train["Store"] == selected_store
]

daily_sales = store_data.groupby("Date")["Sales"].sum()

# Ensure continuous daily frequency
daily_sales = daily_sales.asfreq("D")

# Fill missing dates using forward fill
daily_sales = daily_sales.fillna(method="ffill")



# KPI SECTION

st.subheader(f"Store {selected_store} Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Daily Sales",
        f"{int(daily_sales.mean()):,}"
    )

with col2:
    st.metric(
        "Maximum Daily Sales",
        f"{int(daily_sales.max()):,}"
    )

with col3:
    st.metric(
        "Total Sales",
        f"{int(daily_sales.sum()):,}"
    )

st.write(
    f"Forecast horizon: {forecast_days} days"
)



# TRAIN FORECAST MODEL

model = ExponentialSmoothing(
    daily_sales,
    trend="add",
    seasonal="mul",
    seasonal_periods=7
)

fit = model.fit()

forecast = fit.forecast(forecast_days)



# FORECAST VISUALIZATION

st.subheader("Sales Forecast Visualization")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=daily_sales[-history_days:].index,
        y=daily_sales[-history_days:].values,
        mode="lines",
        name="Historical Sales"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines",
        name="Forecasted Sales"
    )
)

fig.update_layout(
    title=f"Store {selected_store} Sales Forecast",
    xaxis_title="Date",
    yaxis_title="Daily Sales (€)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# FORECAST TABLE

forecast_df = pd.DataFrame({
    "Forecasted Sales":
    forecast.round(0).astype(int)
})

forecast_df["Forecasted Sales"] = (
    forecast_df["Forecasted Sales"]
    .map("{:,}".format)
)

st.subheader("Forecast Values")

st.dataframe(
    forecast_df,
    use_container_width=True
)



# HISTORICAL DATA PREVIEW

st.subheader("Historical Sales Data")

historical_preview = store_data[
    ["Date", "Sales"]
].tail(20)

historical_preview["Sales"] = (
    historical_preview["Sales"]
    .astype(int)
)

st.dataframe(
    historical_preview,
    use_container_width=True
)



# BUSINESS INSIGHTS

st.subheader("Business Insights")

st.markdown(
    f"""
    - Store **{selected_store}** demonstrates recurring weekly sales patterns.
    - Forecasted sales trends may support inventory and staffing planning.
    - Seasonal fluctuations suggest varying customer demand across different periods.
    - Forecasting models can help improve operational planning and resource allocation.
    """
)



