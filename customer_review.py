import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page setup
st.set_page_config(page_title="Interactive Sales Visualizations", layout="wide")
st.markdown("<h1 style='text-align: center; color: #003366;'>📈 Interactive Visualizations from Customer Reviews</h1>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\chira\OneDrive\Documents\ecommerece-analysis-main (1)\ecommerece-analysis-main\customer_reviews_large.csv")

df = load_data()
df.drop_duplicates(inplace=True)

# Label encode for visualization
df['Sentiment_Label'] = df['Sentiment'].astype(str)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating', 'Monthly_Sales'])

# Row 1: Rating & Sentiment
st.markdown("### 🔍 Rating & Sentiment Distribution")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Rating', nbins=5, title="Rating Distribution", color_discrete_sequence=['skyblue'])
    fig1.update_layout(bargap=0.2)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x='Sentiment', title="Sentiment Distribution", color='Sentiment', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Sales vs Rating & Sentiment
st.markdown("### 📊 Sales Patterns")
col3, col4 = st.columns(2)

with col3:
    fig3 = px.box(df, x='Rating', y='Monthly_Sales', color='Sentiment_Label',
                  title="Monthly Sales vs Rating", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.violin(df, y='Monthly_Sales', x='Sentiment_Label', box=True, points='all',
                     title="Sales Distribution by Sentiment", color='Sentiment_Label',
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig4, use_container_width=True)

# Row 3: Top Products by Sales & Review Length
st.markdown("### 🛍️ Product-Based Analysis")
top_products = df.groupby('Product_ID')['Monthly_Sales'].sum().nlargest(10).index
df_top = df[df['Product_ID'].isin(top_products)]

col5, col6 = st.columns(2)

with col5:
    fig5 = px.bar(df_top.groupby('Product_ID')['Monthly_Sales'].sum().reset_index(),
                  x='Product_ID', y='Monthly_Sales', title="Top 10 Products by Sales",
                  color='Monthly_Sales', color_continuous_scale='Teal')
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    df['Review_Length'] = df['Review_Text'].astype(str).apply(len)
    fig6 = px.histogram(df, x='Review_Length', nbins=50, title="Review Length Distribution",
                        color_discrete_sequence=['indianred'])
    st.plotly_chart(fig6, use_container_width=True)
