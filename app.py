import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# ----------------------------------
# App Title & Description
# ----------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

st.divider()

# ----------------------------------
# Upload Data
# ----------------------------------
st.subheader("📁 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")
    st.dataframe(data.head(), use_container_width=True)

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    # ----------------------------------
    # Sidebar Controls (Mandatory)
    # ----------------------------------
    st.sidebar.header("⚙️ Clustering Controls")

    selected_features = st.sidebar.multiselect(
        "Select at least TWO numerical features",
        numeric_cols
    )

    k = st.sidebar.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=3
    )

    random_state = st.sidebar.number_input(
        "Random State (Optional)",
        min_value=0,
        value=42
    )

    run_clustering = st.sidebar.button("🟦 Run Clustering")

    # ----------------------------------
    # Validation
    # ----------------------------------
    if run_clustering:
        if len(selected_features) < 2:
            st.error("Please select **at least two features** for clustering.")
        else:
            # ----------------------------------
            # Data Scaling
            # ----------------------------------
            X = data[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ----------------------------------
            # K-Means
            # ----------------------------------
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            data["Cluster"] = clusters

            st.success("Clustering completed successfully!")

            st.divider()

            # ----------------------------------
            # Visualization Section
            # ----------------------------------
            st.subheader("📊 Cluster Visualization")

            x_feature = selected_features[0]
            y_feature = selected_features[1]

            fig, ax = plt.subplots(figsize=(8, 6))

            scatter = ax.scatter(
                data[x_feature],
                data[y_feature],
                c=data["Cluster"],
                cmap="tab10",
                s=80,
                alpha=0.7
            )

            # Plot cluster centers
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="black",
                s=300,
                marker="X",
                label="Cluster Centers"
            )

            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title("Customer Clusters")
            ax.legend()

            st.pyplot(fig)

            st.divider()

            # ----------------------------------
            # Cluster Summary Table
            # ----------------------------------
            st.subheader("📋 Cluster Summary")

            summary = (
                data.groupby("Cluster")[selected_features]
                .agg(["count", "mean"])
                .round(2)
            )

            summary.columns = [
                f"{col[0]} - {col[1]}" for col in summary.columns
            ]

            st.dataframe(summary, use_container_width=True)

            st.divider()

            # ----------------------------------
            # Business Interpretation
            # ----------------------------------
            st.subheader("💡 Business Interpretation")

            for cluster_id in sorted(data["Cluster"].unique()):
                cluster_data = data[data["Cluster"] == cluster_id]
                avg_values = cluster_data[selected_features].mean()

                dominant_feature = avg_values.idxmax()

                st.markdown(
                    f"""
                    **🟢 Cluster {cluster_id}:**  
                    Customers in this group show higher activity in **{dominant_feature}**,  
                    indicating a similar purchasing pattern and spending behavior.
                    """
                )

            st.divider()

            # ----------------------------------
            # User Guidance Box
            # ----------------------------------
            st.info(
                "Customers in the same cluster exhibit similar purchasing behaviour "
                "and can be targeted with similar business strategies."
            )

else:
    st.info("👆 Upload a dataset to begin customer segmentation.")
