import pandas as pd
import streamlit as st
import plotly.express as px
import umap
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
st.set_page_config(layout="wide")


st.title("Topic embeddings UMAP")

df = pd.read_csv("data/embeddings.csv")
fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['topic'],
                 title="UMAP Projection with Clustering",
                 labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
                 color_continuous_scale=px.colors.sequential.Viridis)


fig.update_layout(
    autosize=True,
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

