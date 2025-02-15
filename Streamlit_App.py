import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load pre-trained models
scaler = joblib.load('Models/scaler.joblib')
pca = joblib.load('Models/pca.joblib')
kmeans = joblib.load('Models/kmeans.joblib')
cluster_names = joblib.load('Models/cluster_names.joblib')

st.title("Customer Segmentation App")
st.sidebar.header("Enter Customer Data")
total_orders = st.sidebar.number_input("Total Orders", 0, 100, 10)
quantity = st.sidebar.number_input("Total Quantity Purchased", 0, 50000, 10000)
unit_price = st.sidebar.number_input("Unit Price", 0.0, 100.0, 20.0)

if st.sidebar.button("Predict Cluster"):
    total_price = quantity * unit_price
    new_customer = pd.DataFrame([[total_orders, quantity, total_price]],
                                columns=['TotalOrders', 'Quantity', 'TotalPrice'])
    scaled_features = scaler.transform(new_customer)
    pca_features = pca.transform(scaled_features)
    predicted_cluster = kmeans.predict(pca_features)[0]
    st.session_state['predicted_cluster'] = predicted_cluster
    st.session_state['distance_to_centroid'] = np.linalg.norm(pca_features[0] - kmeans.cluster_centers_[predicted_cluster])
    st.session_state['pca_features'] = pca_features  # Store PCA features for visualization

# Display prediction results if available
if 'predicted_cluster' in st.session_state:
    st.subheader("Prediction Results")
    st.write(f"**Predicted Cluster:** {cluster_names.get(st.session_state['predicted_cluster'], 'Unknown')}")
    st.write(f"**Distance to Cluster Centroid:** {st.session_state['distance_to_centroid']:.2f}")

if st.checkbox("Show Cluster Visualizations and Calculate Silhouette Score"):
    customer_data = pd.read_csv('Data/processed_customer_data.csv')
    pca_customer_data = pca.transform(scaler.transform(customer_data[['TotalOrders', 'Quantity', 'TotalPrice']]))
    cluster_labels = customer_data['Cluster'].values

    silhouette_avg = silhouette_score(pca_customer_data, cluster_labels)
    st.write(f"**Silhouette Score for the model:** {silhouette_avg:.2f}")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue']  # Adjust color count if necessary
    for i, cluster in enumerate(np.unique(cluster_labels)):
        ax.scatter(pca_customer_data[cluster_labels == cluster, 0],
                   pca_customer_data[cluster_labels == cluster, 1],
                   c=colors[i % len(colors)], label=f'Cluster {cluster}: {cluster_names[cluster]}', alpha=0.5)

    # Highlight the newly predicted customer
    if 'pca_features' in st.session_state:
        ax.scatter(st.session_state['pca_features'][:, 0], st.session_state['pca_features'][:, 1], c='magenta', edgecolors='black', s=200, marker='*', label='New Customer')

    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')
    ax.set_title('Customer Segmentation with Named Clusters')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    ax.grid(True)

    # Show the plot in Streamlit
    st.pyplot(fig)
