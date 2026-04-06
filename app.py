"""
app.py — DriftScope: Global Knowledge Evolution Analyzer

Big Data Stack:
  - Apache Spark (PySpark)  reads ALL papers from the data_lake for every search
  - Hadoop LocalFileSystem  provides the underlying storage abstraction for Spark
  - BERT (sentence-transformers)  embeds matched papers on-the-fly
  - NumPy  computes cosine similarity between temporal centroids

Search architecture (on-demand, not pre-filtered):
  User types keyword
    → Spark scans EVERY paper in data_lake for each target year
    → Spark SQL filter: keyword in title OR abstract OR categories
    → Matching papers collected to driver node
    → BERT encodes matching abstracts
    → Cosine-similarity drift scored across years
    → Dashboard updated
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["JAVA_HOME"]   = r"C:\Program Files\Java\jdk-18.0.2.1"

BASE_DIR      = r"c:\Users\Afham Faiyaz Ahmad\Desktop\big data\DriftScope-main"
DATA_LAKE     = os.path.join(BASE_DIR, "data_lake")
DRIFT_CSV     = os.path.join(BASE_DIR, "drift_results.csv")
DUMMY_CSV     = os.path.join(BASE_DIR, "dummy_drift_data.csv")
TARGET_YEARS  = [2010, 2015, 2020, 2023, 2025]

# ── Page config (original) ────────────────────────────────────────────────────
st.set_page_config(page_title="DriftScope Analyzer", layout="wide")
st.title("DriftScope: Global Knowledge Evolution Analyzer")

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Starting Apache Spark session...")
def get_spark() -> SparkSession:
    """
    One Spark session shared across all Streamlit reruns.
    Hadoop LocalFileSystem is the storage layer for reading data_lake Parquet.
    """
    spark = (
        SparkSession.builder
        .appName("DriftScope_Dashboard")
        .master("local[*]")                         # all CPU cores
        .config("spark.driver.memory",    "8g")
        .config("spark.sql.shuffle.partitions", "8")
        # Force pure-Java Hadoop FS (avoids Windows NativeIO for reads)
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource(show_spinner="Loading BERT model...")
def get_bert_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Drift computation ─────────────────────────────────────────────────────────
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1, v2 = v1.astype(np.float64), v2.astype(np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0


@st.cache_data(show_spinner="Scanning all papers with Apache Spark + computing BERT drift...")
def compute_drift_all_papers(keyword: str) -> pd.DataFrame:
    """
    Scans EVERY paper in the data_lake using Apache Spark,
    filters by keyword across title, abstract, AND categories,
    then embeds matching abstracts with BERT and scores drift.

    Big Data path:
      PySpark reads data_lake Parquet (Hadoop FS)
      → distributed keyword filter (Spark SQL)
      → collect matched rows to driver
      → BERT encode (sentence-transformers)
      → NumPy cosine-similarity centroid drift
    """
    spark = get_spark()
    model = get_bert_model()
    kw = keyword.strip().lower()

    records      = []
    baseline_vec = None
    prev_vec     = None

    for yr in TARGET_YEARS:
        parquet_path = os.path.join(DATA_LAKE, f"year={yr}", "part-00000.parquet")

        if not os.path.exists(parquet_path):
            # Parquet missing → record as 0 so it still appears on the graph
            records.append({
                "Topic": keyword, "Year": yr,
                "Semantic_Drift_Score":    0.0,
                "Semantic_Drift_Velocity": 0.0,
                "Fact_Stability_Index":    "No Data",
                "Semantic_Shock":          False,
                "PaperCount":              0,
                "TotalPapersScanned":      0,
            })
            continue

        # ── Spark: read ALL papers for this year ──────────────────────────
        df = spark.read.parquet(parquet_path)
        total_scanned = df.count()

        # ── Spark SQL: distributed keyword filter across 3 columns ────────
        matched = df.filter(
            lower(col("title")).contains(kw)      |
            lower(col("abstract")).contains(kw)   |
            lower(col("categories")).contains(kw)
        )
        count = matched.count()

        if count == 0:
            # Concept not yet coined this year → SDS = 0.0 (appears on graph)
            # Do NOT reset prev_vec — maintain velocity continuity
            records.append({
                "Topic": keyword, "Year": yr,
                "Semantic_Drift_Score":    0.0,
                "Semantic_Drift_Velocity": 0.0,
                "Fact_Stability_Index":    "No Data",
                "Semantic_Shock":          False,
                "PaperCount":              0,
                "TotalPapersScanned":      total_scanned,
            })
            continue

        # ── Collect to driver and embed with BERT ─────────────────────────
        # Use a max of 250 abstracts per year to compute the centroid quickly
        papers_pd = matched.select("abstract").limit(250).toPandas()
        abstracts = papers_pd["abstract"].dropna().str.strip()
        abstracts = abstracts[abstracts != ""].tolist()

        if not abstracts:
            records.append({
                "Topic": keyword, "Year": yr,
                "Semantic_Drift_Score":    0.0,
                "Semantic_Drift_Velocity": 0.0,
                "Fact_Stability_Index":    "No Data",
                "Semantic_Shock":          False,
                "PaperCount":              0,
                "TotalPapersScanned":      total_scanned,
            })
            continue

        embeddings = model.encode(
            abstracts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        centroid = embeddings.mean(axis=0)

        if baseline_vec is None:
            baseline_vec = centroid   # anchor = first year the concept appears

        sds  = 1.0 - cosine_similarity(baseline_vec, centroid)
        # Velocity vs previous year that actually HAD papers
        v_sd = (1.0 - cosine_similarity(prev_vec, centroid)) \
               if prev_vec is not None else 0.0
        fsi  = "Stable"   if sds < 0.15 else \
               "Evolving" if sds < 0.50 else "Volatile"
        shock = bool(v_sd > 0.10)

        records.append({
            "Topic":                   keyword,
            "Year":                    yr,
            "Semantic_Drift_Score":    round(sds,  4),
            "Semantic_Drift_Velocity": round(v_sd, 4),
            "Fact_Stability_Index":    fsi,
            "Semantic_Shock":          shock,
            "PaperCount":              count,
            "TotalPapersScanned":      total_scanned,
        })
        prev_vec = centroid

    return pd.DataFrame(records)


# ── Load pre-scored CSV (for cross-topic comparison only) ─────────────────────
@st.cache_data
def load_prescored() -> pd.DataFrame | None:
    if os.path.exists(DRIFT_CSV):
        return pd.read_csv(DRIFT_CSV)
    if os.path.exists(DUMMY_CSV):
        return pd.read_csv(DUMMY_CSV)
    return None


# ── Sidebar (original look, search bar instead of dropdown) ──────────────────
st.sidebar.header("Knowledge Query")
search_query = st.sidebar.text_input(
    "Search a Concept to Analyze:",
    value="",
    placeholder="e.g. agentic AI, deep learning, climate...",
)
search_btn = st.sidebar.button("Analyze", width="stretch")

data_lake_ok = os.path.exists(DATA_LAKE)
st.sidebar.markdown("---")
st.sidebar.caption(f"{'✅' if data_lake_ok else '❌'} data_lake (Spark / Hadoop)")
st.sidebar.caption(f"{'✅' if os.path.exists(DRIFT_CSV) else '⚠️'} drift_results.csv")


# ── Main logic ────────────────────────────────────────────────────────────────
prescored_df = load_prescored()

# Default: show first pre-scored topic on load
if not search_query and not search_btn:
    if prescored_df is not None:
        default_topic = prescored_df["Topic"].dropna().iloc[0]
        topic_data    = prescored_df[prescored_df["Topic"] == default_topic] \
                            .sort_values("Year")
        selected_topic = default_topic
        scanned_info   = None
    else:
        st.info("Type a keyword in the sidebar to begin analysis.")
        st.stop()

elif search_query:
    if not data_lake_ok:
        st.error("data_lake/ not found. Run data_ingestion.py first.")
        st.stop()

    topic_data     = compute_drift_all_papers(search_query)
    selected_topic = search_query
    scanned_info   = topic_data["TotalPapersScanned"].sum() \
                     if "TotalPapersScanned" in topic_data.columns else None
else:
    st.warning("Please type a keyword and click Analyze.")
    st.stop()


# ── Guard: truly no papers at all across every year ─────────────────────────
# (SDS=0 is valid for years where the concept hasn't appeared yet)
if topic_data["PaperCount"].sum() == 0:
    st.warning(
        f"No papers found for **\"{selected_topic}\"** "
        f"in any of the target years ({', '.join(map(str, TARGET_YEARS))}).\n\n"
        "Try a broader term (e.g. 'neural' instead of 'neural network')."
    )
    if scanned_info:
        st.caption(f"Spark scanned {scanned_info:,} total papers across all years.")
    st.stop()

# valid_rows: years that actually have papers (used for metrics display)
valid_rows = topic_data[topic_data["PaperCount"] > 0]

topic_data = topic_data.sort_values("Year")

if scanned_info:
    matched_total = int(topic_data["PaperCount"].sum())
    st.caption(
        f"Spark scanned **{scanned_info:,}** papers · "
        f"matched **{matched_total:,}** for \"{selected_topic}\""
    )

# ── Metric cards — use latest year that actually has papers ──────────────────
st.subheader(f"Semantic Metrics for: {selected_topic}")

latest_data = valid_rows.iloc[-1]  # most recent year with ≥1 matching paper
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Semantic Drift Score",
              f"{latest_data['Semantic_Drift_Score']:.4f}")
with col2:
    st.metric("Drift Velocity (V_sd)",
              f"{latest_data['Semantic_Drift_Velocity']:.4f}")
with col3:
    st.metric("Fact Stability Index", latest_data["Fact_Stability_Index"])
with col4:
    shock = bool(latest_data["Semantic_Shock"])
    st.metric("Shock Status",
              "⚠️ SHOCK DETECTED" if shock else "Stable")

# ── Drift Visualization — use ALL years including zeros ──────────────────────
st.markdown("---")
st.subheader("Semantic Drift Over Time (2010 - 2025)")

# topic_data has ALL 5 years; years with no papers show SDS=0 on the graph
fig = px.line(
    topic_data,
    x="Year",
    y="Semantic_Drift_Score",
    markers=True,
    title=f"Evolution Trajectory of '{selected_topic}'",
)
fig.update_layout(yaxis_range=[0, 1])
st.plotly_chart(fig, width="stretch")

# ── Velocity bar chart — all years, zero where no papers ─────────────────────
st.subheader("Drift Velocity (Year-over-Year Change)")
fig2 = px.bar(
    topic_data,
    x="Year",
    y="Semantic_Drift_Velocity",
    title=f"Semantic Shock Events for '{selected_topic}'",
    color="Semantic_Shock",
    color_discrete_map={True: "#EF4444", False: "#10B981"},
)
st.plotly_chart(fig2, width="stretch")

# ── Cross-topic comparison (original, uses pre-scored CSV) ───────────────────
if prescored_df is not None:
    st.markdown("---")
    st.subheader("Cross-Topic Drift Comparison (Pre-Scored Topics)")
    latest_all = (
        prescored_df[prescored_df["Year"] == prescored_df["Year"].max()]
        .dropna(subset=["Semantic_Drift_Score"])
        .sort_values("Semantic_Drift_Score", ascending=True)
    )
    fig3 = px.bar(
        latest_all,
        x="Semantic_Drift_Score",
        y="Topic",
        orientation="h",
        title="Comparative Semantic Drift Score (Latest Year)",
        color="Semantic_Drift_Score",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig3, width="stretch")

# ── Paper count table ─────────────────────────────────────────────────────────
if "PaperCount" in topic_data.columns:
    st.markdown("---")
    st.subheader("Paper Count by Year")
    cols_to_show = ["Year", "PaperCount"]
    if "TotalPapersScanned" in topic_data.columns:
        cols_to_show.append("TotalPapersScanned")
    tbl = topic_data[cols_to_show].set_index("Year")
    st.dataframe(tbl, width="stretch")