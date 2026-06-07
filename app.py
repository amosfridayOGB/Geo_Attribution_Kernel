import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Geo-Attribution Matrix Kernel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("🔬 Geo-Attribution Multi-Touch Matrix Kernel")
st.markdown("### Advanced Spatial Attribution & Channel Optimization Engine")
st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Attribution Controls")

lambda_decay = st.sidebar.slider(
    "Spatial Decay Constant (λ)",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01
)

st.sidebar.markdown("### Channel Weights")

w_awareness = st.sidebar.slider("Meta Awareness", 0.1, 2.0, 0.3, 0.1)
w_conversion = st.sidebar.slider("Meta Conversion", 0.1, 2.0, 0.8, 0.1)
w_whatsapp = st.sidebar.slider("WhatsApp Lead Trigger", 0.1, 2.0, 1.2, 0.1)

channel_weights = {
    "meta_awareness_ad": w_awareness,
    "meta_conversion_ad": w_conversion,
    "whatsapp_lead_trigger": w_whatsapp
}

# =====================================================
# DATA FUNCTIONS
# =====================================================

DATA_PATH = os.path.join("core_logic", "simulated_touchpoints.csv")

@st.cache_data
def load_default_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    # Synced seamlessly with generate_data.py advanced geometric framework
    np.random.seed(42)
    n = 10000

    channels = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    channel_map = {0: "meta_awareness_ad", 1: "meta_conversion_ad", 2: "whatsapp_lead_trigger"}
    utm_sources = [channel_map[channel] for channel in channels]

    interactions = np.random.geometric(p=0.4, size=n)
    distances = np.clip(np.random.normal(loc=15.0, scale=8.0, size=n), 0.1, 50.0)

    center_lat, center_lon = 6.5244, 3.3792
    latitudes = center_lat + np.random.normal(loc=0, scale=0.15, size=n)
    longitudes = center_lon + np.random.normal(loc=0, scale=0.15, size=n)

    return pd.DataFrame({
        "latitude": np.round(latitudes, 6),
        "longitude": np.round(longitudes, 6),
        "distance_km": np.round(distances, 2),
        "utm_source": utm_sources,
        "interaction_count": interactions
    })

# =====================================================
# FILE UPLOAD
# =====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_dataset()

# =====================================================
# VALIDATION
# =====================================================

required_columns = ["latitude", "longitude", "distance_km", "utm_source", "interaction_count"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# =====================================================
# ATTRIBUTION ENGINE
# =====================================================

distance_array = df["distance_km"].to_numpy()
interaction_array = df["interaction_count"].to_numpy()
channel_array = df["utm_source"].to_numpy()

weight_vector = np.vectorize(lambda x: channel_weights.get(x, 1.0))(channel_array)
spatial_decay = np.exp(-lambda_decay * distance_array)
attribution_scores = interaction_array * weight_vector * spatial_decay

df["attribution_score"] = attribution_scores

# =====================================================
# KPI SECTION
# =====================================================

st.markdown("## 📈 Performance Overview")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Records", f"{len(df):,}")
with k2:
    st.metric("Total Attribution", f"{attribution_scores.sum():.2f}")
with k3:
    st.metric("Average Score", f"{attribution_scores.mean():.4f}")
with k4:
    st.metric("Maximum Score", f"{attribution_scores.max():.2f}")

st.markdown("---")

# =====================================================
# MAP + CHANNEL CHART
# =====================================================

left, right = st.columns([2, 1])

with left:
    st.subheader("🗺️ Geospatial Attribution Map")
    fig_map = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        color="attribution_score",
        size="interaction_count",
        zoom=10,
        size_max=15,
        map_style="carto-darkmatter",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_map, width="stretch")

with right:
    st.subheader("📊 Channel Attribution")
    channel_summary = df.groupby("utm_source")["attribution_score"].sum().reset_index()
    fig_bar = px.bar(channel_summary, x="utm_source", y="attribution_score", color="utm_source")
    st.plotly_chart(fig_bar, width="stretch")

# =====================================================
# DISTRIBUTION ANALYSIS
# =====================================================

st.markdown("---")
st.subheader("📉 Attribution Score Distribution")
histogram = px.histogram(df, x="attribution_score", nbins=50)
st.plotly_chart(histogram, width="stretch")

# =====================================================
# CHANNEL PERFORMANCE TABLE
# =====================================================

st.markdown("---")
st.subheader("🏆 Channel Performance Ranking")
ranking = df.groupby("utm_source").agg(
    Total_Score=("attribution_score", "sum"),
    Average_Score=("attribution_score", "mean"),
    Total_Interactions=("interaction_count", "sum"),
    Avg_Distance=("distance_km", "mean")
).sort_values("Total_Score", ascending=False)

st.dataframe(ranking, width="stretch")

# =====================================================
# TOP RECORDS
# =====================================================

st.markdown("---")
st.subheader("🔥 Highest Attribution Records")
top_records = df.sort_values("attribution_score", ascending=False).head(25)
st.dataframe(top_records, width="stretch")

# =====================================================
# DOWNLOAD RESULTS
# =====================================================

st.markdown("---")
st.subheader("⬇️ Export Results")
csv_output = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Scored Dataset",
    data=csv_output,
    file_name="geo_attribution_results.csv",
    mime="text/csv"
)

# =====================================================
# RAW DATA
# =====================================================

with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df.head(100), width="stretch")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("Geo-Attribution Matrix Kernel • Spatial Decay Attribution Analytics")
