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

@st.cache_data
def load_review_data():
    df = pd.read_csv("customer_reviews_large.csv")
    df.drop_duplicates(inplace=True)
    df['Sentiment_Label'] = df['Sentiment'].astype(str)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating', 'Monthly_Sales'])
    return df

# ------------------------ CLV DASHBOARD ------------------------
def render_clv_dashboard():
    df = load_clv_data()
    st.title("💰 Customer Lifetime Value Dashboard")
    st.image("Screenshot 2025-03-21 121514.png", caption="Customer Behavior Overview", use_column_width=True)

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

    st.markdown("### 🔍 CLV Distribution, Correlation & Age Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(px.histogram(df, x="CLV", nbins=50, title="CLV Distribution"), use_container_width=True)
    with col2:
        corr = df[["Age", "Customer_Tenure", "Total_Spending", "Purchase_Frequency", "Profit_Margin", "CLV"]].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap"), use_container_width=True)
    with col3:
        st.plotly_chart(px.scatter(df, x="Age", y="CLV", title="Age vs CLV"), use_container_width=True)

    st.markdown("### 📈 Purchase Frequency, Feature Importance & Segmentation")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.plotly_chart(px.scatter(df, x="Purchase_Frequency", y="CLV", title="Purchase Frequency vs CLV"), use_container_width=True)

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
        st.plotly_chart(px.bar(importance, orientation='h', title="Feature Importance (Random Forest)"), use_container_width=True)

    with col6:
        seg_features = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        X_scaled = StandardScaler().fit_transform(df[seg_features])
        df['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
        st.plotly_chart(px.scatter(df, x='Income', y='MntWines', color='Cluster', title="Customer Segmentation (KMeans)"), use_container_width=True)

# ------------------------ ASSOCIATION RULES ------------------------
def render_product_pair_advisor():
    rules = load_rules()
    st.title("🧠 Product Pair Recommender")
    st.markdown("Use smart product pairing insights to boost cross-sell performance.")

    with st.sidebar:
        st.header("🔧 Recommendation Filters")
        min_success = st.slider("🎯 Minimum Success Rate (%)", 1, 100, 40)
        min_frequency = st.slider("📊 Minimum Frequency (%)", 0.0, 5.0, 0.5, step=0.1, format="%.2f%%")

    filtered = rules[(rules['success_rate'] >= min_success) & (rules['combination_frequency'] >= min_frequency)]
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
        st.plotly_chart(fig, use_container_width=True)

# ------------------------ CUSTOMER REVIEWS ANALYSIS ------------------------
def render_customer_reviews():
    df = load_review_data()
    st.title("📝 Customer Review Insights")
    st.markdown("Visual exploration of customer sentiment, ratings, and their impact on sales.")

    # Section 1: Rating and Sentiment Overview
    st.subheader("📊 Ratings & Sentiment Distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x='Rating', nbins=5, title="⭐ Rating Distribution", color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='Sentiment', title="🧠 Sentiment Categories", color='Sentiment', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

    # Section 2: Sentiment Impact on Sales
    st.subheader("💹 Sales Patterns by Rating and Sentiment")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(
            df, x='Rating', y='Monthly_Sales', color='Sentiment_Label',
            title="📦 Monthly Sales by Rating and Sentiment",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.violin(
            df, y='Monthly_Sales', x='Sentiment_Label', box=True, points='all',
            title="🎯 Sales by Sentiment", color='Sentiment_Label',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Section 3: Product and Review Insights
    st.subheader("📈 Product Popularity & Review Patterns")

    df['Review_Length'] = df['Review_Text'].astype(str).apply(len)

    col5, col6 = st.columns(2)
    with col5:
        top_products = df.groupby('Product_ID')['Monthly_Sales'].sum().nlargest(10).index
        df_top = df[df['Product_ID'].isin(top_products)]
        fig5 = px.bar(
            df_top.groupby('Product_ID')['Monthly_Sales'].sum().reset_index(),
            x='Product_ID', y='Monthly_Sales',
            title="🏆 Top 10 Products by Sales",
            color='Monthly_Sales',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.histogram(
            df, x='Review_Length', nbins=50,
            title="📝 Review Length Distribution",
            color_discrete_sequence=['#d62728']
        )
        st.plotly_chart(fig6, use_container_width=True)

# ------------------------ MAIN ------------------------
def main():
    st.set_page_config(page_title="Ebuss Insights", layout="wide")
    st.sidebar.title("📊 Dashboard Navigation")
    section = st.sidebar.radio("Choose a section:", ["CLV Dashboard", "Product Pair Recommender", "Customer Reviews Analysis"])

    if section == "CLV Dashboard":
        render_clv_dashboard()
    elif section == "Product Pair Recommender":
        render_product_pair_advisor()
    elif section == "Customer Reviews Analysis":
        render_customer_reviews()

if __name__ == "__main__":
    main()
