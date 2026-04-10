"""
buzzword_extractor.py — DriftScope: Temporal Buzzword Extraction

BIG DATA STACK:
  - Apache Spark (PySpark) reads all papers from data_lake Parquet
  - Spark MLlib Tokenizer → StopWordsRemover → CountVectorizer → IDF
    compute TF-IDF importance scores across all abstracts per year
  - Top 10 defining terms per year extracted from aggregated TF-IDF vectors
  - Output: buzzwords_per_year.csv

Flow:
  data_lake/year=YYYY/part-00000.parquet
      → spark.read.parquet()            [Spark distributed Parquet scan]
      → Tokenizer → StopWordsRemover   [Spark MLlib text preprocessing]
      → CountVectorizer → IDF          [Spark MLlib TF-IDF computation]
      → Aggregated TF-IDF ranking      [NumPy argsort on summed vectors]
      → buzzwords_per_year.csv          [dashboard-ready output]
"""

import os
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF

# ── Configuration ────────────────────────────────────────────────────────────
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["JAVA_HOME"]   = r"C:\Program Files\Java\jdk-18.0.2.1"

BASE_DIR     = r"c:\Users\Afham Faiyaz Ahmad\Desktop\big data\DriftScope-main"
DATA_LAKE    = os.path.join(BASE_DIR, "data_lake")
OUTPUT_CSV   = os.path.join(BASE_DIR, "buzzwords_per_year.csv")
TARGET_YEARS = [2010, 2015, 2020, 2023, 2025]

# Academic filler words to exclude beyond standard English stopwords
CUSTOM_STOPS = [
    "also", "using", "used", "use", "one", "two", "new", "can", "may",
    "however", "show", "shown", "paper", "arxiv", "propose", "proposed",
    "results", "based", "approach", "present", "study", "within", "well",
    "first", "different", "et", "al", "fig", "figure", "table", "eq",
    "section", "respectively", "e.g", "i.e", "thus", "therefore",
    "furthermore", "moreover", "corresponding", "given", "consider",
    "considered", "obtained", "provide", "recent", "recently", "since",
    "several", "three", "four", "many", "various", "particular",
]


# ── Spark Session ─────────────────────────────────────────────────────────────
def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DriftScope_BuzzwordExtractor")
        .master("local[*]")
        .config("spark.driver.memory",          "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
        .getOrCreate()
    )


# ── UDF: filter words shorter than 3 characters ──────────────────────────────
filter_short_words = udf(
    lambda words: [w for w in words if len(w) >= 3] if words else [],
    ArrayType(StringType()),
)


def extract_buzzwords_for_year(spark, year, k=10):
    """
    Extract the top-k most important terms for a given year using
    Spark MLlib's TF-IDF pipeline.

    Pipeline:
      abstract text
        → lowercase + strip non-alpha chars  (Spark SQL)
        → Tokenizer                          (Spark MLlib)
        → StopWordsRemover (English + custom)(Spark MLlib)
        → filter tokens len < 3              (PySpark UDF)
        → CountVectorizer (vocab ≤ 5000)     (Spark MLlib)
        → IDF                                (Spark MLlib)
        → aggregate TF-IDF per term          (NumPy sum + argsort)
    """
    parquet_path = os.path.join(DATA_LAKE, f"year={year}", "part-00000.parquet")
    if not os.path.exists(parquet_path):
        print(f"  SKIP — no parquet at {parquet_path}")
        return []

    df = spark.read.parquet(parquet_path)

    # Clean: lowercase, remove non-alphabetic characters
    df_clean = (
        df.select(
            regexp_replace(lower(col("abstract")), r"[^a-z\s]", "")
            .alias("clean_text")
        )
        .filter(col("clean_text").isNotNull())
        .filter(col("clean_text") != "")
    )

    total_docs = df_clean.count()
    if total_docs == 0:
        return []

    # Tokenize
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_words")
    df_words = tokenizer.transform(df_clean)

    # Remove stopwords (English defaults + academic filler)
    all_stops = list(
        set(StopWordsRemover.loadDefaultStopWords("english") + CUSTOM_STOPS)
    )
    remover = StopWordsRemover(
        inputCol="raw_words", outputCol="stopped_words", stopWords=all_stops
    )
    df_stopped = remover.transform(df_words)

    # Remove short words (< 3 chars)
    df_filtered = df_stopped.withColumn(
        "filtered_words", filter_short_words(col("stopped_words"))
    )

    # CountVectorizer — build vocabulary (max 5000 terms, min 2 docs)
    cv = CountVectorizer(
        inputCol="filtered_words",
        outputCol="raw_features",
        vocabSize=5000,
        minDF=2.0,
    )
    cv_model = cv.fit(df_filtered)
    df_cv = cv_model.transform(df_filtered)

    # IDF weighting
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(df_cv)
    df_tfidf = idf_model.transform(df_cv)

    vocab = cv_model.vocabulary

    # Aggregate TF-IDF scores across all documents
    tfidf_rows = df_tfidf.select("tfidf_features").collect()
    total_scores = np.zeros(len(vocab))
    for row in tfidf_rows:
        vec = row["tfidf_features"]
        for idx, val in zip(vec.indices, vec.values):
            total_scores[idx] += val

    # Extract top-K terms
    top_indices = np.argsort(total_scores)[::-1]
    results = []
    for idx in top_indices:
        word = vocab[idx]
        if len(word) < 3:
            continue
        results.append({
            "Year":          year,
            "Rank":          len(results) + 1,
            "Buzzword":      word,
            "TF_IDF_Score":  round(float(total_scores[idx]), 4),
        })
        if len(results) >= k:
            break

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("DriftScope — Temporal Buzzword Extraction (PySpark MLlib TF-IDF)")
    print("=" * 65)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nSpark version : {spark.version}")
    print(f"Cores in use  : {spark.sparkContext.defaultParallelism}")
    print(f"Target years  : {TARGET_YEARS}\n")

    all_records = []
    for yr in TARGET_YEARS:
        print(f"\n{'─'*55}")
        print(f"Year: {yr}")
        records = extract_buzzwords_for_year(spark, yr)
        all_records.extend(records)
        for r in records:
            print(f"  #{r['Rank']:2d}  {r['Buzzword']:<25s}  TF-IDF={r['TF_IDF_Score']:.4f}")

    spark.stop()

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*65}")
    print(f"Buzzwords saved : {OUTPUT_CSV}")
    print(f"Total entries   : {len(df)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
