"""
data_ingestion.py — DriftScope Stage 1

BIG DATA STACK:
  - Apache Spark (PySpark) reads and processes the 5 GB arXiv JSON
    using its distributed, in-memory DataFrame engine (Stage 0 = 39 partitions)
  - Hadoop HDFS-compatible API governs all path resolution
  - PyArrow handles the final Parquet write per-year on disk
    (bypasses Hadoop's NativeIO Windows file-commit bug while
     keeping all heavy processing inside Spark's JVM)

Flow:
  arXiv JSON (5 GB)
      → spark.read.json()          [Spark distributed JSON scan]
      → select / withColumn / filter  [Spark DataFrame transforms]
      → .toPandas() per year       [collect to driver, one year at a time]
      → PyArrow parquet write      [fast columnar disk write]
      → data_lake/year=YYYY/part-00000.parquet
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, to_date

# ── Configuration ────────────────────────────────────────────────────────────
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["JAVA_HOME"]   = r"C:\Program Files\Java\jdk-18.0.2.1"

BASE_DIR     = r"c:\Users\Afham Faiyaz Ahmad\Desktop\big data\DriftScope-main"
INPUT_PATH   = r"c:\Users\Afham Faiyaz Ahmad\Desktop\big data\arxiv-metadata-oai-snapshot.json"
OUTPUT_DIR   = os.path.join(BASE_DIR, "data_lake")
TARGET_YEARS = [2010, 2015, 2020, 2023, 2025]

PARQUET_SCHEMA = pa.schema([
    pa.field("id",          pa.string()),
    pa.field("title",       pa.string()),
    pa.field("abstract",    pa.string()),
    pa.field("categories",  pa.string()),
    pa.field("update_date", pa.string()),
    pa.field("year",        pa.int32()),
])

# ── Spark Session ─────────────────────────────────────────────────────────────
def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DriftScope_DataIngestion")
        .master("local[*]")                                   # all CPU cores
        .config("spark.driver.memory",              "8g")
        .config("spark.executor.memory",            "8g")
        .config("spark.sql.shuffle.partitions",     "8")
        # Hadoop/Kerberos config — pure-Java paths, no native IO required
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.file.impl",
                "org.apache.hadoop.fs.local.LocalFs")
        .getOrCreate()
    )

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("DriftScope — Stage 1: Data Ingestion (PySpark Big Data)")
    print("=" * 65)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nSpark version  : {spark.version}")
    print(f"Hadoop version : {spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}")
    print(f"Cores in use   : {spark.sparkContext.defaultParallelism}")
    print(f"\nInput  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Target years: {TARGET_YEARS}\n")

    # ── Stage 0: Spark reads the 5 GB JSON (distributed across partitions) ──
    print("Reading arXiv JSON with Spark (distributed scan)...")
    df_raw = spark.read.json(INPUT_PATH)

    # ── Stage 1: Spark DataFrame transforms ──────────────────────────────────
    df = (
        df_raw
        .select("id", "title", "abstract", "categories", "update_date")
        .withColumn("date", to_date(col("update_date"), "yyyy-MM-dd"))
        .withColumn("year", year(col("date")))
        .filter(col("year").isin(TARGET_YEARS))
        .drop("date")
    )

    total = df.count()
    print(f"Spark filtered {total:,} matching records across target years.\n")

    # ── Stage 2: collect one year at a time → PyArrow write ──────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for yr in TARGET_YEARS:
        print(f"Writing year={yr}  ...", end=" ", flush=True)

        year_df: pd.DataFrame = (
            df.filter(col("year") == yr)
              .select("id", "title", "abstract", "categories", "update_date", "year")
              .toPandas()
        )
        year_df["year"] = year_df["year"].astype("int32")

        out_dir  = os.path.join(OUTPUT_DIR, f"year={yr}")
        out_path = os.path.join(out_dir, "part-00000.parquet")
        os.makedirs(out_dir, exist_ok=True)

        tbl = pa.Table.from_pandas(year_df, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(tbl, out_path, compression="snappy")

        size_mb = os.path.getsize(out_path) / 1e6
        print(f"{len(year_df):,} rows → {size_mb:.1f} MB")

    spark.stop()

    print("\n" + "=" * 65)
    print("Stage 1 complete — data_lake/ is ready for the NLP pipeline.")
    print("=" * 65)


if __name__ == "__main__":
    main()