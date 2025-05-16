import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# ------------------------ DATA LOADERS ------------------------
@st.cache_data
def load_clv_data():
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

@st.cache_data
def load_rules():
    rules = pd.read_csv("associationRules.csv")
    rules['success_rate'] = (rules['confidence'] * 100).round(1)
    rules['combination_frequency'] = (rules['support'] * 100).round(3)
    return rules

# ------------------------ CLV VISUALS ------------------------
def render_clv_dashboard():
    df = load_clv_data()
    st.title("💰 Customer Lifetime Value Dashboard")

    st.image("Screenshot 2025-03-21 121514.png", caption="Customer Behavior Overview", use_column_width=True)

    # Row 1: Boxplot of Numerical Features using Matplotlib
    st.subheader("📦 Boxplot of Numerical Features")
    numerical_cols = [
        "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth", "Z_CostContact", "Z_Revenue"
    ]

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[idx])
        axes[idx].set_title(col)

    for j in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # Row 2
    st.markdown("### 🔍 CLV Distribution, Correlation & Age Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.histogram(df, x="CLV", nbins=50, title="CLV Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        corr = df[["Age", "Customer_Tenure", "Total_Spending", "Purchase_Frequency", "Profit_Margin", "CLV"]].corr()
        fig2 = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.scatter(df, x="Age", y="CLV", title="Age vs CLV")
        st.plotly_chart(fig3, use_container_width=True)

    # Row 3
    st.markdown("### 📈 Purchase Frequency, Feature Importance & Segmentation")
    col4, col5, col6 = st.columns(3)

    with col4:
        fig4 = px.scatter(df, x="Purchase_Frequency", y="CLV", title="Purchase Frequency vs CLV")
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        features = ['Income', 'Recency', 'Total_Spending', 'NumDealsPurchases', 'NumWebPurchases', 'NumStorePurchases']
        X = df[features]
        y = df["CLV"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        importance = pd.Series(model.feature_importances_, index=features).sort_values()
        fig5 = px.bar(importance, orientation='h', title="Feature Importance (Random Forest)")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        seg_features = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        X_scaled = StandardScaler().fit_transform(df[seg_features])
        df['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
        fig6 = px.scatter(df, x='Income', y='MntWines', color='Cluster', title="Customer Segmentation (KMeans)", color_continuous_scale='viridis')
        st.plotly_chart(fig6, use_container_width=True)

# ------------------------ ASSOCIATION RULES ------------------------
def render_product_pair_advisor():
    rules = load_rules()
    st.title("🧠 Product Pair Recommender")
    st.markdown("Use smart product pairing insights to boost cross-sell performance.")

    with st.sidebar:
        st.header("🔧 Recommendation Filters")
        min_success = st.slider("🎯 Minimum Success Rate (%)", 1, 100, 40)
        min_frequency = st.slider("📊 Minimum Frequency (%)", 0.0, 5.0, 0.5, step=0.1, format="%.2f%%")

    filtered = rules[
        (rules['success_rate'] >= min_success) &
        (rules['combination_frequency'] >= min_frequency)
    ].sort_values('success_rate', ascending=False)

    st.markdown(f"### 📦 Showing {len(filtered)} Product Recommendations")
    if not filtered.empty:
        for _, row in filtered.iterrows():
            st.markdown(f"""
                #### 🛒 Bought: {row['antecedents']}
                **➕ Recommend:** {row['consequents']}  
                🎯 **Success Rate:** {row['success_rate']}%  
                📊 **Frequency:** {row['combination_frequency']}%
            """)
    else:
        st.warning("No recommendations meet current filter criteria.")

    if not filtered.empty:
        st.markdown("### 🔍 Visual Explorer")
        fig = px.scatter(
            filtered,
            x='combination_frequency',
            y='success_rate',
            size='lift',
            color='lift',
            hover_name='antecedents',
            hover_data={'consequents': True},
            title="Lift vs Frequency vs Confidence"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------ MAIN ------------------------
def main():
    st.set_page_config(page_title="Ebuss CLV + Product Pairing", layout="wide")
    st.sidebar.title("📊 Navigation")
    section = st.sidebar.radio("Choose a section:", ["CLV Dashboard", "Product Pair Recommender"])

    if section == "CLV Dashboard":
        render_clv_dashboard()
    elif section == "Product Pair Recommender":
        render_product_pair_advisor()

if __name__ == "__main__":
    main()
