import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob  # Alternative to sentimentr
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Banking Data Modeling Dashboard", layout="wide")

# Title
st.title("Banking Data Modeling Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        sep = st.radio("Separator", [",", ";", "\t"], index=0)
        header = st.checkbox("Header", value=True)
        
        # Read data
        df = pd.read_csv(uploaded_file, sep=sep, header=0 if header else None)
        
        # Clean column names
        df.columns = df.columns.str.replace('[^a-zA-Z0-9]', '').str.lower()
        
        st.header("Model Settings")
        k_clusters = st.slider("Number of Clusters (K-means):", 2, 5, 3)
        text_col = st.selectbox("Text Column for Sentiment Analysis:", df.columns)
        retention_col = st.selectbox("Retention Column (0/1) for Logistic Regression:", df.columns)
        esg_col = st.selectbox("ESG Score Column:", df.columns)
        
        analyze = st.button("Run Analysis", type="primary")

# Main content
if uploaded_file is not None:
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Preview", 
        "Sentiment Analysis", 
        "Retention Model", 
        "Customer Segments"
    ])
    
    with tab1:
        st.dataframe(df.head())
    
    if analyze:
        # Sentiment Analysis (simplified)
        with tab2:
            st.subheader("Sentiment Analysis")
            
            # Simple sentiment analysis with TextBlob
            df['sentiment_score'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            df['sentiment_label'] = pd.cut(df['sentiment_score'], 
                                         bins=[-1, -0.1, 0.1, 1],
                                         labels=["Negative", "Neutral", "Positive"])
            
            # Plot
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='sentiment_label', 
                         order=["Negative", "Neutral", "Positive"],
                         palette={"Negative": "tomato", "Neutral": "gray", "Positive": "steelblue"},
                         ax=ax)
            ax.set_title("Distribution of Sentiment Labels")
            st.pyplot(fig)
            
            # Show results
            st.dataframe(df[[text_col, 'sentiment_score', 'sentiment_label']])
        
        # Logistic Regression Model
        with tab3:
            st.subheader("Retention Model")
            
            try:
                # Prepare data
                X = df[[esg_col]].dropna()
                X = StandardScaler().fit_transform(X)
                y = df.loc[X.index, retention_col].astype(int)
                
                # Fit model
                model = LogisticRegression().fit(X, y)
                
                # Show coefficients
                st.write("Model Coefficients:")
                st.write(f"Intercept: {model.intercept_[0]:.4f}")
                st.write(f"{esg_col}: {model.coef_[0][0]:.4f}")
                
                # Plot
                fig, ax = plt.subplots()
                sns.regplot(x=df[esg_col], y=df[retention_col].astype(int),
                            logistic=True, scatter_kws={'alpha':0.3},
                            line_kws={'color':'steelblue'}, ax=ax)
                ax.set_title("ESG Score vs. Retention Probability")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in logistic regression: {str(e)}")
        
        # K-means Clustering
        with tab4:
            st.subheader("Customer Segments")
            
            try:
                # Select and scale numeric features
                num_cols = [esg_col, 'rating', 'usefulcount']  # Adjust as needed
                num_data = df[num_cols].dropna()
                scaled_data = StandardScaler().fit_transform(num_data)
                
                # Cluster
                kmeans = KMeans(n_clusters=k_clusters, random_state=123).fit(scaled_data)
                
                # Add clusters back to original data
                df['cluster'] = np.nan
                df.loc[num_data.index, 'cluster'] = kmeans.labels_
                
                # Plot clusters
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x='rating', y=esg_col, 
                              hue='cluster', palette='viridis', alpha=0.7, ax=ax)
                ax.set_title("Customer Segments by Rating and ESG Score")
                st.pyplot(fig)
                
                # Silhouette score
                silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
                st.write(f"Silhouette Score: {silhouette_avg:.2f}")
                
                # Cluster statistics
                cluster_stats = df.groupby('cluster').agg({
                    esg_col: 'mean',
                    'rating': 'mean',
                    'cluster': 'count'
                }).rename(columns={'cluster': 'n_customers'})
                st.dataframe(cluster_stats)
                
            except Exception as e:
                st.error(f"Error in clustering: {str(e)}")