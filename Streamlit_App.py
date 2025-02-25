import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# Load pre-trained models
scaler = joblib.load('Models/scaler.joblib')
pca = joblib.load('Models/pca.joblib')
kmeans = joblib.load('Models/kmeans.joblib')
cluster_names = joblib.load('Models/cluster_names.joblib')

data = pd.read_csv('Data/Online Retail.csv')

customer_data = pd.read_csv('Data/cleaned_data.csv')

# Filter out products with UnitPrice = 0
filtered_data = data[data['UnitPrice'] != 0]

# Sidebar tabs for EDA
st.sidebar.title("Navigation")
tabs = ["Make Predictions", "Exploratory Data Analysis (EDA)", "Scaling Numerical Values", "Model Building", "Model Accuracy", "Save Model"]
selected_tab = st.sidebar.radio("Go to", tabs)

# Home tab (main functionality)
if selected_tab == "Make Predictions":
    # Main app title
    st.title("Customer Segmentation App")

    # Sidebar for customer order data
    st.sidebar.header("Enter Customer Order Data")

    # Dropdown to select a product (excluding products with UnitPrice = 0)
    product_list = filtered_data['Description'].unique()  # Get unique product descriptions
    selected_product = st.sidebar.selectbox("Select Product", product_list)

    # Fetch the unit price for the selected product
    unit_price = filtered_data[filtered_data['Description'] == selected_product]['UnitPrice'].values[0]

    # Display the unit price
    st.sidebar.write(f"**Unit Price for {selected_product}:** ${unit_price}")

    # Input for quantity
    quantity = st.sidebar.number_input("Total Quantity Purchased", 0, 100000000, 10000)
        
    # Input for total_orders (optional, if needed)
    total_orders = st.sidebar.number_input("Total Orders", 0, 1000000, 10)
       
    # Calculate total_price dynamically
    total_price = unit_price * quantity * total_orders

    # Display the calculated total_price
    st.sidebar.write(f"**Calculated Total Invoice Price:** ${total_price}")

    if st.sidebar.button("Predict Customer Segment"):
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

        silhouette_avg = silhouette_score(pca_customer_data, cluster_labels) * 100
        st.write(f"**Silhouette Score (Accuracy) for the model:** {silhouette_avg:.2f}%")

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

# EDA Tabs
elif selected_tab == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("#### Import Libraries")
    st.code("""
    # imporitng libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import PowerTransformer
    """, language='python')

    st.write("#### Read Data")
    st.code("""
    # reading data
    data = pd.read_csv('Online Retail.csv')
    data.head(5)
    """, language='python')
    
    st.dataframe(data.head(5))
    
    st.write("#### Understand Data")
    st.code("""
    # understanding data
    data.info()
    """, language='python')
    
    # Capture the output of data.info()
    buffer = StringIO()
    data.info(buf=buffer)
    info_output = buffer.getvalue()
    st.text(info_output)  # Display as plain text
    
    st.write("#### Clean Data")
    st.code("""
    # cleaning data
    data.isnull().sum()
    """, language='python')
    
    st.write(data.isnull().sum())

    st.code("""
    # cleaning data
    cleaned_data = data.dropna(subset=['CustomerID', 'Description'])

    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])

    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']

    customer_data = cleaned_data.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Number of unique transactions (orders) per customer
        'Quantity': 'sum',       # Total quantity of products purchased by each customer
        'TotalPrice': 'sum'      # Total spending by each customer
    }).rename(columns={'InvoiceNo': 'TotalOrders'})

    customer_data.reset_index(inplace=True)

    customer_data.head()
    """, language='python')
    
    cleaned_data = data.dropna(subset=['CustomerID', 'Description'])

    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])

    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']

    customer_data = cleaned_data.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Number of unique transactions (orders) per customer
        'Quantity': 'sum',       # Total quantity of products purchased by each customer
        'TotalPrice': 'sum'      # Total spending by each customer
    }).rename(columns={'InvoiceNo': 'TotalOrders'})

    customer_data.reset_index(inplace=True)

    customer_data.head()
    
    st.dataframe(customer_data.head(5))
    
elif selected_tab == "Scaling Numerical Values":
    st.title("Scaling Numerical Values")
    st.write("#### Implementing Robust Scaler")
    st.code("""
    # Implementing Robust Scaler
    scaler = RobustScaler()
    scaled_customer_data = scaler.fit_transform(customer_data[['TotalOrders', 'Quantity', 'TotalPrice']])
    """, language='python')
    
    scaler = RobustScaler()
    scaled_customer_data = scaler.fit_transform(customer_data[['TotalOrders', 'Quantity', 'TotalPrice']])

elif selected_tab == "Model Building":
    st.title("K-Means Clustering Model for Customer Segmentation")
    st.write("#### Elbow method to find optimal number of clusters")
    st.code("""
    # implementing elbow method
    inertia = []
    cluster_range = range(1, 11)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_customer_data)
        inertia.append(kmeans.inertia_)

    # Plotting the elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.xticks(cluster_range)
    plt.grid(True)
    plt.show()
    """, language='python')
    
    # Ensure customer_data is properly defined
    cleaned_data = data.dropna(subset=['CustomerID', 'Description'])
    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])
    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']
    customer_data = cleaned_data.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Number of unique transactions (orders) per customer
        'Quantity': 'sum',       # Total quantity of products purchased by each customer
        'TotalPrice': 'sum'      # Total spending by each customer
    }).rename(columns={'InvoiceNo': 'TotalOrders'})
    customer_data.reset_index(inplace=True)
    
    # Scale the data
    scaler = RobustScaler()
    scaled_customer_data = scaler.fit_transform(customer_data[['TotalOrders', 'Quantity', 'TotalPrice']])
    
    # Calculate inertia for different cluster numbers
    inertia = []
    cluster_range = range(1, 11)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_customer_data)
        inertia.append(kmeans.inertia_)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cluster_range, inertia, marker='o')
    ax.set_title('Elbow Method For Optimal k')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia (Sum of squared distances)')
    ax.set_xticks(cluster_range)
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # PCA Section
    st.write("#### Principle Component Analysis (PCA)")
    st.code("""
    # implementing PCA
    # Applying PCA to reduce dimensions while capturing as much variance as possible
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
    pca_customer_data = pca.fit_transform(scaled_customer_data)

    # Plotting the PCA result to visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_customer_data[:, 0], pca_customer_data[:, 1], alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
    plt.title('PCA Result (2 Components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

    # Variance explained by each component
    pca.explained_variance_ratio_
    """, language='python')

    # Create the PCA plot
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
    pca_customer_data = pca.fit_transform(scaled_customer_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pca_customer_data[:, 0], pca_customer_data[:, 1], alpha=0.5)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
    ax.set_title('PCA Result (2 Components)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Display the explained variance ratio
    st.write("**Explained Variance Ratio:**")
    explained_variance = pca.explained_variance_ratio_
    for i, variance in enumerate(explained_variance, start=1):
        st.write(f"PCA{i}: {variance:.4f}")
    
    # K-Means Clustering Model
    st.write("#### K-Means Clustering Model")
    st.code("""
    # implementing k-means clustering model
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(pca_customer_data)

    # Assign clusters back to the customer data
    customer_data['Cluster'] = kmeans.labels_

    # Calculate mean values for each cluster to understand their characteristics
    cluster_analysis = customer_data.groupby('Cluster').agg({
        'TotalOrders': 'mean',
        'Quantity': 'mean',
        'TotalPrice': 'mean'
    }).rename(columns={
        'TotalOrders': 'Average Total Orders',
        'Quantity': 'Average Quantity',
        'TotalPrice': 'Average Total Price'
    })
    
    print(cluster_analysis)
    
    """, language='python')
    
    # implementing k-means clustering model
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(pca_customer_data)

    # Assign clusters back to the customer data
    customer_data['Cluster'] = kmeans.labels_

    # Calculate mean values for each cluster to understand their characteristics
    cluster_analysis = customer_data.groupby('Cluster').agg({
        'TotalOrders': 'mean',
        'Quantity': 'mean',
        'TotalPrice': 'mean'
    }).rename(columns={
        'TotalOrders': 'Average Total Orders',
        'Quantity': 'Average Quantity',
        'TotalPrice': 'Average Total Price'
    })

    st.dataframe(cluster_analysis)
    
    st.write("#### Assigning Cluster Names")
    st.code("""
    # Assign cluster names
    cluster_names = {
        0: 'Low Spending Customers',
        1: 'High Value Customers',
        2: 'Moderate Spending Customers'
    }
    customer_data['Customer Segment'] = customer_data['Cluster'].map(cluster_names)

    # Unique colors for clusters
    colors = np.array(['red', 'green', 'blue'])

    plt.figure(figsize=(10, 6))
    for cluster in np.unique(customer_data['Cluster']):
        # Select only data rows where cluster is equal to the current loop cluster
        cluster_data = pca_customer_data[customer_data['Cluster'] == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[cluster], label=f'Cluster {cluster}: {cluster_names[cluster]}', alpha=0.5)

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
    plt.title('Customer Segmentation with Named Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()                          
    """, language='python')

    # Assign cluster names
    cluster_names = {
        0: 'Low Spending Customers',
        1: 'High Value Customers',
        2: 'Moderate Spending Customers'
    }
    customer_data['Customer Segment'] = customer_data['Cluster'].map(cluster_names)

    # Unique colors for clusters
    colors = np.array(['red', 'green', 'blue'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in np.unique(customer_data['Cluster']):
        # Select only data rows where cluster is equal to the current loop cluster
        cluster_data = pca_customer_data[customer_data['Cluster'] == cluster]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[cluster], label=f'Cluster {cluster}: {cluster_names[cluster]}', alpha=0.5)

    # Plot centroids
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')

    # Add labels, title, and legend
    ax.set_title('Customer Segmentation with Named Clusters')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)
    
elif selected_tab == "Model Accuracy":
    st.title("Model Accuracy")
    st.write("#### Silhouette Score for Model Accuracy")
    st.code("""
    # silhouette score
    from sklearn.metrics import silhouette_score

    # Calculate the silhouette score for the K-means clustering
    silhouette_avg = silhouette_score(pca_customer_data, kmeans.labels_)

    print(silhouette_avg)
    """, language='python')
    
    # Ensure customer_data is properly defined
    cleaned_data = data.dropna(subset=['CustomerID', 'Description'])
    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])
    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']
    customer_data = cleaned_data.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Number of unique transactions (orders) per customer
        'Quantity': 'sum',       # Total quantity of products purchased by each customer
        'TotalPrice': 'sum'      # Total spending by each customer
    }).rename(columns={'InvoiceNo': 'TotalOrders'})
    customer_data.reset_index(inplace=True)
    
    # Scale the data
    scaler = RobustScaler()
    scaled_customer_data = scaler.fit_transform(customer_data[['TotalOrders', 'Quantity', 'TotalPrice']])
    
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
    pca_customer_data = pca.fit_transform(scaled_customer_data)
    
    # silhouette score
    from sklearn.metrics import silhouette_score

    # Calculate the silhouette score for the K-means clustering
    silhouette_avg = silhouette_score(pca_customer_data, kmeans.labels_)
    silhouette_avg = round(silhouette_avg * 100, 2)
    
    st.write(f"Model accuracy: {silhouette_avg}%")

elif selected_tab == "Save Model":
    st.title("Save Model")
    st.code("""
    import joblib

    # Save the trained models
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(pca, 'pca.joblib')
    joblib.dump(kmeans, 'kmeans.joblib')

    # Save the cluster labels dictionary
    joblib.dump(cluster_names, 'cluster_names.joblib')

    print("Models and cluster names saved successfully!")
    """, language='python')