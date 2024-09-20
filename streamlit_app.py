import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")


st.title("Topic embeddings for primary source documents")

df = pd.read_csv("data/topic_clustering.csv")

fig = px.scatter(
    df, 
    x='x', 
    y='y', 
    color='cluster', 
    hover_data=['topic'],
    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
    color_continuous_scale=px.colors.qualitative.Light24
)

fig.update_layout(
    autosize=True,
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

