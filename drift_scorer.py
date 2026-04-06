"""
drift_scorer.py — DriftScope Stage 3: Semantic Drift Scoring

BIG DATA STACK:
  - Apache Spark (PySpark) loads the embedding Parquet files and performs
    keyword filtering as a distributed DataFrame operation
  - Spark RDD .map()/.collect() pulls the matching embedding vectors
    to the driver node for centroid computation
  - NumPy computes cosine similarity between 384-dim BERT centroid vectors

Flow:
  embeddings_lake/year=YYYY/
      → spark.read.parquet()            [Spark distributed Parquet scan]
      → DataFrame.filter() by keyword   [Spark SQL predicate pushdown]
      → RDD.map().collect()             [gather vectors to driver]
      → numpy centroid + cosine sim     [analytical computation]
      → drift_results.csv               [dashboard-ready output]
"""

import os
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower

# ── Configuration ─────────────────────────────────────────────────────────────
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["JAVA_HOME"]   = r"C:\Program Files\Java\jdk-18.0.2.1"

BASE_DIR       = r"c:\Users\Afham Faiyaz Ahmad\Desktop\big data\DriftScope-main"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings_lake")
OUTPUT_CSV     = os.path.join(BASE_DIR, "drift_results.csv")
TARGET_YEARS   = [2010, 2015, 2020, 2023, 2025]

# Topics to pre-score → display_name: arXiv keyword
TOPICS = {
    "AI":             "artificial intelligence",
    "Blockchain":     "blockchain",
    "Climate Change": "climate",
    "Transformers":   "transformer",
    "COVID":          "covid",
}

# ── Spark Session ──────────────────────────────────────────────────────────────
def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DriftScope_DriftScorer")
        .master("local[*]")
        .config("spark.driver.memory",          "8g")
        .config("spark.executor.memory",        "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
        .getOrCreate()
    )

# ── Analytics ─────────────────────────────────────────────────────────────────
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1, v2 = v1.astype(np.float64), v2.astype(np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0


def get_topic_centroid_spark(spark: SparkSession, year: int, keyword: str):
    """
    Use Spark to read the embedding Parquet for `year`,
    filter rows whose title/categories contain `keyword`,
    then collect the embedding vectors to the driver and return
    the mean centroid vector + paper count.
    """
    parquet_path = os.path.join(EMBEDDINGS_DIR, f"year={year}", "part-00000.parquet")
    if not os.path.exists(parquet_path):
        return None, 0

    # Spark distributed Parquet read
    df = spark.read.parquet(parquet_path)

    kw = keyword.lower()

    # Spark SQL predicate filtering (runs distributed)
    filtered = df.filter(
        lower(col("title")).contains(kw) |
        lower(col("categories")).contains(kw)
    )

    count = filtered.count()
    if count == 0:
        return None, 0

    # Collect embedding vectors from all workers to the driver
    vectors = (
        filtered.select("embedding_vector")
                .rdd
                .map(lambda r: r[0])
                .filter(lambda v: v is not None)
                .collect()
    )

    if not vectors:
        return None, 0

    centroid = np.array(vectors, dtype=np.float32).mean(axis=0)
    return centroid, count


def compute_drift(spark: SparkSession, display_name: str, keyword: str) -> list[dict]:
    """Compute Semantic Drift Score, Velocity, FSI and Shock for one topic."""
    records = []
    prev_vec     = None
    baseline_vec = None

    for yr in TARGET_YEARS:
        print(f"    [{yr}] Spark scan...", end=" ", flush=True)
        centroid, count = get_topic_centroid_spark(spark, yr, keyword)

        if centroid is None:
            print(f"no papers for '{keyword}'")
            records.append({
                "Topic": display_name, "Year": yr,
                "Semantic_Drift_Score":    None,
                "Semantic_Drift_Velocity": None,
                "Fact_Stability_Index":    "Unknown",
                "Semantic_Shock":          False,
                "PaperCount":              0,
            })
            continue

        if baseline_vec is None:
            baseline_vec = centroid  # anchor = first year with data

        sds  = 1.0 - cosine_similarity(baseline_vec, centroid)
        v_sd = (1.0 - cosine_similarity(prev_vec, centroid)) if prev_vec is not None else 0.0
        fsi  = "Stable" if sds < 0.15 else ("Evolving" if sds < 0.50 else "Volatile")
        shock = bool(v_sd > 0.10)

        print(f"SDS={sds:.4f}  V_sd={v_sd:.4f}  FSI={fsi}  shock={shock}  papers={count}")
        records.append({
            "Topic":                   display_name,
            "Year":                    yr,
            "Semantic_Drift_Score":    round(sds,  4),
            "Semantic_Drift_Velocity": round(v_sd, 4),
            "Fact_Stability_Index":    fsi,
            "Semantic_Shock":          shock,
            "PaperCount":              count,
        })
        prev_vec = centroid

    return records


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("DriftScope — Stage 3: Drift Scoring (PySpark + NumPy)")
    print("=" * 65)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nSpark version : {spark.version}")
    print(f"Cores in use  : {spark.sparkContext.defaultParallelism}")
    print(f"Topics        : {list(TOPICS)}\n")

    all_records = []
    for display_name, keyword in TOPICS.items():
        print(f"\nTopic: {display_name}  (keyword: '{keyword}')")
        records = compute_drift(spark, display_name, keyword)
        all_records.extend(records)

    spark.stop()

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*65}")
    print(f"Drift results saved: {OUTPUT_CSV}")
    print(f"Rows: {len(results_df)}")
    print()
    print(results_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
