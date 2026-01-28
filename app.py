import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
        color: black;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Content wrapper */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Section headers */
    h2, h3 {
        color: #1e293b;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: black;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    /* Cluster cards */
    .cluster-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .cluster-card:hover {
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-3px);
    }
    
    .cluster-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    .uploadedFile {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown('<h1>📊 Wholesale Customer Segmentation</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced K-Means clustering to identify and analyze customer purchasing patterns with precision</p>', unsafe_allow_html=True)

# ---------------------------
# Upload dataset
# ---------------------------
st.markdown("### 📁 Data Upload")

uploaded_file = st.file_uploader(
    "Upload your wholesale customer dataset (CSV format)",
    type=["csv"],
    help="Upload a CSV file containing wholesale customer data with spending features"
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Quick stats in metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{len(data):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{len(data.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Spending</div>
            <div class="metric-value">${data.select_dtypes(include=[np.number]).mean().mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Quality</div>
            <div class="metric-value">{(1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------
    # Dataset Preview
    # ---------------------------
    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True, height=300)

    # ---------------------------
    # Feature Selection
    # ---------------------------
    spending_features = [
        'Fresh', 'Milk', 'Grocery',
        'Frozen', 'Detergents_Paper', 'Delicassen'
    ]

    st.markdown("---")
    st.markdown("### 🎯 Spending Features Analysis")
    
    st.markdown("""
    <div class="info-box">
        <strong>Selected Features:</strong> Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
        <br><strong>Method:</strong> StandardScaler normalization followed by K-Means clustering
    </div>
    """, unsafe_allow_html=True)

    X = data[spending_features]

    # Feature distribution
    with st.expander("📊 View Feature Distributions"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.patch.set_facecolor('white')
        
        for idx, feature in enumerate(spending_features):
            ax = axes[idx // 3, idx % 3]
            ax.hist(data[feature], bins=30, color='#667eea', alpha=0.7, edgecolor='white')
            ax.set_title(feature, fontsize=12, fontweight='bold', color='#1e293b')
            ax.set_xlabel('Spending', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------
    # Scaling
    # ---------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("✅ Data successfully standardized using StandardScaler")

    st.markdown("---")

    # ---------------------------
    # Clustering Configuration
    # ---------------------------
    st.markdown("### ⚙️ Clustering Configuration")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        k = st.slider(
            "Select number of clusters (K)",
            min_value=2,
            max_value=8,
            value=3,
            help="Choose the optimal number of customer segments"
        )
    
    with col2:
        st.markdown(f"""
        <div class="info-box" style="margin-top: 0;">
            <strong>Selected K:</strong> {k}
            <br><strong>Algorithm:</strong> K-Means
            <br><strong>Random State:</strong> 42
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------
    # K-Means Clustering
    # ---------------------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    data["Cluster"] = clusters

    st.success(f"✅ K-Means clustering completed successfully with {k} clusters")

    # Cluster distribution
    cluster_counts = data["Cluster"].value_counts().sort_index()
    
    st.markdown("#### Cluster Distribution")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        
        colors = sns.color_palette("husl", k)
        bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='white', linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
        ax.set_title('Customer Distribution Across Clusters', fontsize=14, fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Cluster Sizes:**")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(data)) * 100
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.25rem 0; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 8px;">
                <strong>Cluster {cluster_id}:</strong> {count} ({percentage:.1f}%)
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------
    # Cluster Visualization
    # ---------------------------
    st.markdown("### 📈 Interactive Cluster Visualization")

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox("X-axis feature", spending_features, index=2)
    with col2:
        y_feature = st.selectbox("Y-axis feature", spending_features, index=4)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Scatter plot with custom styling
    scatter = sns.scatterplot(
        x=data[x_feature],
        y=data[y_feature],
        hue=data["Cluster"],
        palette="husl",
        s=100,
        alpha=0.7,
        edgecolor='white',
        linewidth=1.5,
        ax=ax
    )

    # Plot centroids
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    x_index = spending_features.index(x_feature)
    y_index = spending_features.index(y_feature)

    ax.scatter(
        centers[:, x_index],
        centers[:, y_index],
        c="black",
        s=500,
        marker="X",
        label="Centroids",
        edgecolor='white',
        linewidth=2,
        zorder=5
    )

    ax.set_xlabel(x_feature, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_feature, fontsize=13, fontweight='bold')
    ax.set_title(f'Customer Clusters: {x_feature} vs {y_feature}', fontsize=15, fontweight='bold', pad=20)
    ax.legend(title='Cluster', title_fontsize=11, fontsize=10, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------
    # Cluster Profiling
    # ---------------------------
    st.markdown("### 📊 Detailed Cluster Profiles")
    
    profile = data.groupby("Cluster")[spending_features].mean().round(2)
    
    # Styled dataframe
    st.dataframe(
        profile.style.background_gradient(cmap='YlOrRd', axis=1).format("${:,.2f}"),
        use_container_width=True,
        height=250
    )

    # Heatmap visualization
    st.markdown("#### 🔥 Cluster Spending Heatmap")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    sns.heatmap(
        profile,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Average Spending'},
        linewidths=2,
        linecolor='white',
        ax=ax
    )
    
    ax.set_title('Average Spending by Cluster and Category', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Spending Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------
    # Business Insights
    # ---------------------------
    st.markdown("### 💡 Strategic Business Insights")

    for cluster_id in profile.index:
        dominant = profile.loc[cluster_id].idxmax()
        dominant_value = profile.loc[cluster_id].max()
        cluster_size = len(data[data['Cluster'] == cluster_id])
        
        st.markdown(f"""
        <div class="cluster-card">
            <div class="cluster-header">🎯 Cluster {cluster_id} - {cluster_size} Customers</div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Dominant Category:** {dominant} (${dominant_value:,.2f} avg)")
        
        # Business recommendations based on dominant category
        if dominant in ["Grocery", "Detergents_Paper"]:
            st.markdown("""
            <div class="insight-box">
                <strong>🛒 Retail-Focused Segment</strong><br>
                • Implement volume-based discount programs<br>
                • Prioritize inventory for high-turnover products<br>
                • Consider loyalty programs for repeat purchases<br>
                • Focus on consistent supply chain management
            </div>
            """, unsafe_allow_html=True)
        elif dominant in ["Fresh", "Frozen", "Delicassen"]:
            st.markdown("""
            <div class="insight-box">
                <strong>🍽️ HoReCa (Hotel/Restaurant/Café) Segment</strong><br>
                • Offer premium quality product bundles<br>
                • Implement frequent delivery schedules<br>
                • Provide specialized freshness guarantees<br>
                • Create seasonal menu-based promotions
            </div>
            """, unsafe_allow_html=True)
        elif dominant == "Milk":
            st.markdown("""
            <div class="insight-box">
                <strong>🥛 Dairy-Focused Segment</strong><br>
                • Cross-sell complementary dairy products<br>
                • Offer subscription-based delivery models<br>
                • Promote health and nutrition benefits<br>
                • Ensure cold chain excellence
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
                <strong>📦 Balanced/Growth Opportunity Segment</strong><br>
                • Deploy targeted promotional campaigns<br>
                • Increase engagement through personalized offers<br>
                • Analyze purchase patterns for upsell opportunities<br>
                • Consider product bundling strategies
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------
    # Final Dataset Export
    # ---------------------------
    st.markdown("### 📁 Clustered Dataset")
    
    st.dataframe(
        data.style.apply(
            lambda x: ['background-color: #e0f2fe' if x.name % 2 == 0 else '' for i in x],
            axis=1
        ),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Clustered Data",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv",
    )

else:
    st.markdown("""
    <div class="info-box" style="margin-top: 2rem; text-align: center; padding: 3rem;">
        <h3>👆 Get Started</h3>
        <p>Upload your wholesale customer dataset (CSV) to begin the segmentation analysis</p>
        <p style="color: #64748b; font-size: 0.9rem; margin-top: 1rem;">
            Expected columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p style="font-size: 0.9rem;">Built with Streamlit • Powered by K-Means Clustering</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">Customer Segmentation Analytics Platform</p>
</div>
""", unsafe_allow_html=True)