import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("customer_data.csv")
    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns=["Education", "Marital_Status"], drop_first=True)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    current_year = datetime.now().year
    df["Age"] = current_year - df["Year_Birth"]
    df["Customer_Tenure"] = (datetime.now() - df["Dt_Customer"]).dt.days

    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["Total_Spending"] = df[spending_cols].sum(axis=1)

    purchase_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Purchase_Frequency"] = df[purchase_cols].sum(axis=1)

    df["Profit_Margin"] = df["Total_Spending"] * 0.3
    df["CLV"] = df["Profit_Margin"] * df["Purchase_Frequency"] * df["Customer_Tenure"]

    return df

# Visualizations
def draw_boxplots(df):
    st.subheader("📦 Boxplot of Numerical Features")
    numerical_cols = [
        "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth", "Z_CostContact", "Z_Revenue"
    ]
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    axs = axs.flatten()
    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axs[i], color="skyblue")
        axs[i].set_title(col, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

def clv_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["CLV"], bins=50, kde=True, ax=ax, color="salmon")
    ax.set_title("CLV Distribution")
    return fig

def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[["Age", "Customer_Tenure", "Total_Spending", "Purchase_Frequency", "Profit_Margin", "CLV"]].corr(),
                annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig

def age_vs_clv(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df["Age"], y=df["CLV"], alpha=0.6, ax=ax, color="green")
    ax.set_title("Age vs CLV")
    return fig

def purchase_freq_vs_clv(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df["Purchase_Frequency"], y=df["CLV"], alpha=0.6, ax=ax, color="purple")
    ax.set_title("Purchase Frequency vs CLV")
    return fig

def feature_importance_plot(df):
    features = ['Income', 'Recency', 'Total_Spending', 'NumDealsPurchases', 'NumWebPurchases', 'NumStorePurchases']
    X = df[features]
    y = df["CLV"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importance = pd.Series(model.feature_importances_, index=features)
    fig, ax = plt.subplots(figsize=(6, 4))
    importance.sort_values().plot(kind='barh', ax=ax, color="orange")
    ax.set_title("Feature Importance (Random Forest)")
    return fig

def customer_segmentation_plot(df):
    features = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df['Income'], y=df['MntWines'], hue=df['Cluster'], palette="viridis", ax=ax)
    ax.set_title("Customer Segmentation (KMeans)")
    return fig

# Main Streamlit App
def main():
    st.set_page_config(page_title="💰 CLV Visual Dashboard", layout="wide")
    st.title("📊 Customer Lifetime Value (CLV) Dashboard")

    df = load_data()

    # Row 1: Boxplot
    draw_boxplots(df)

    # Row 2: CLV Dist | Correlation | Age vs CLV
    st.markdown("### 🔍 CLV Distribution, Correlation & Age Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(clv_distribution(df))
    with col2:
        st.pyplot(correlation_heatmap(df))
    with col3:
        st.pyplot(age_vs_clv(df))

    # Row 3: Purchase Freq vs CLV | Feature Importance | Segmentation
    st.markdown("### 🔍 Purchase Patterns, Importance & Segmentation")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.pyplot(purchase_freq_vs_clv(df))
    with col5:
        st.pyplot(feature_importance_plot(df))
    with col6:
        st.pyplot(customer_segmentation_plot(df))

if __name__ == "__main__":
    main()
