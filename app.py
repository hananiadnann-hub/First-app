import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page config
st.set_page_config(page_title="Combined Banking Dashboard", layout="wide")

# Title
st.title("Combined Banking Data Analysis Dashboard")

# Function to load data
@st.cache_data
def load_data():
    # Load both datasets (in production, you'd use file_uploader or store in cloud)
    try:
        data1 = pd.read_excel("data/sustainable_banking.xlsx")  # Update path
        data2 = pd.read_excel("data/bank_reviews.xlsx")        # Update path
        
        # Clean column names
        data1.columns = data1.columns.str.replace('[^a-zA-Z0-9]', '').str.lower()
        data2.columns = data2.columns.str.replace('[^a-zA-Z0-9]', '').str.lower()
        
        # Merge datasets (adjust join logic as needed)
        # Here we assume there's a common 'bank_id' or similar column
        if 'bank_id' in data1.columns and 'bank_id' in data2.columns:
            combined = pd.merge(data1, data2, on='bank_id', how='inner')
        else:
            # If no common column, just concatenate
            combined = pd.concat([data1, data2], axis=0)
            
        return combined
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

if not df.empty:
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Settings")
        
        # Select columns for analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        k_clusters = st.slider("Number of Clusters (K-means):", 2, 5, 3)
        text_col = st.selectbox("Text Column for Sentiment Analysis:", text_cols)
        
        if len(numeric_cols) >= 2:
            retention_col = st.selectbox("Retention Column (0/1):", numeric_cols)
            esg_col = st.selectbox("ESG Score Column:", numeric_cols)
            cluster_feature1 = st.selectbox("First Clustering Feature:", numeric_cols, index=0)
            cluster_feature2 = st.selectbox("Second Clustering Feature:", numeric_cols, index=min(1, len(numeric_cols)-1))
        else:
            st.warning("Not enough numeric columns for analysis")
        
        analyze = st.button("Run Analysis", type="primary")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview", 
        "Sentiment Analysis", 
        "Retention Model", 
        "Customer Segments"
    ])
    
    with tab1:
        st.subheader("Combined Data Preview")
        st.write(f"Shape: {df.shape} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
        
        # Show basic stats
        st.subheader("Numeric Column Statistics")
        st.dataframe(df.describe())
        
        # Show missing values
        st.subheader("Missing Values")
        missing = df.isnull().sum().to_frame(name="Missing Count")
        st.dataframe(missing[missing["Missing Count"] > 0])
    
    if analyze:
        # Sentiment Analysis
        with tab2:
            st.subheader("Sentiment Analysis")
            
            # Calculate sentiment
            df['sentiment_score'] = df[text_col].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else np.nan
            )
            df['sentiment_label'] = pd.cut(
                df['sentiment_score'],
                bins=[-1, -0.1, 0.1, 1],
                labels=["Negative", "Neutral", "Positive"]
            )
            
            # Plot distribution
            fig1 = px.histogram(
                df, x='sentiment_label', 
                category_orders={"sentiment_label": ["Negative", "Neutral", "Positive"]},
                color='sentiment_label',
                color_discrete_map={
                    "Negative": "tomato",
                    "Neutral": "gray",
                    "Positive": "steelblue"
                },
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show most positive/negative reviews
            st.subheader("Sample Reviews by Sentiment")
            pos_sample = df[df['sentiment_label'] == "Positive"].sample(3)[[text_col, 'sentiment_score']]
            neg_sample = df[df['sentiment_label'] == "Negative"].sample(3)[[text_col, 'sentiment_score']]
            
            st.write("**Positive Reviews**")
            st.dataframe(pos_sample)
            st.write("**Negative Reviews**")
            st.dataframe(neg_sample)
        
        # Retention Model
        with tab3:
            st.subheader("Retention Prediction Model")
            
            try:
                # Prepare data
                retention_data = df[[esg_col, retention_col]].dropna()
                X = retention_data[[esg_col]]
                y = retention_data[retention_col].astype(int)
                
                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit model
                model = LogisticRegression().fit(X_scaled, y)
                
                # Show results
                st.write("**Model Summary**")
                st.write(f"- Intercept: {model.intercept_[0]:.4f}")
                st.write(f"- Coefficient for {esg_col}: {model.coef_[0][0]:.4f}")
                
                # Plot relationship
                fig2 = px.scatter(
                    retention_data,
                    x=esg_col,
                    y=retention_col,
                    trendline="lowess",
                    title=f"{esg_col} vs Retention"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show predictions
                retention_data['predicted_prob'] = model.predict_proba(X_scaled)[:, 1]
                st.write("**Sample Predictions**")
                st.dataframe(retention_data.sample(5))
                
            except Exception as e:
                st.error(f"Could not build retention model: {str(e)}")
        
        # Customer Segmentation
        with tab4:
            st.subheader("Customer Segmentation")
            
            try:
                # Select features for clustering
                cluster_data = df[[cluster_feature1, cluster_feature2]].dropna()
                
                # Scale data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Cluster
                kmeans = KMeans(n_clusters=k_clusters, random_state=42).fit(scaled_data)
                cluster_data['cluster'] = kmeans.labels_
                
                # Plot clusters
                fig3 = px.scatter(
                    cluster_data,
                    x=cluster_feature1,
                    y=cluster_feature2,
                    color='cluster',
                    title="Customer Segments",
                    hover_data=cluster_data.columns
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Silhouette score
                silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
                st.write(f"Silhouette Score: {silhouette_avg:.2f} (Higher is better)")
                
                # Cluster characteristics
                st.subheader("Cluster Characteristics")
                cluster_stats = cluster_data.groupby('cluster').agg({
                    cluster_feature1: ['mean', 'std'],
                    cluster_feature2: ['mean', 'std'],
                })
                st.dataframe(cluster_stats)
                
            except Exception as e:
                st.error(f"Could not perform clustering: {str(e)}")
else:
    st.warning("No data loaded. Please check your data files.")