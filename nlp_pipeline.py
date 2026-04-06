"""
nlp_pipeline.py — DriftScope Stage 2: BERT Embeddings via Spark NLP

Big Data Stack:
  - sparknlp.start()            launches Spark with Spark NLP jars loaded
  - DocumentAssembler           Spark NLP: wraps raw text into NLP Document type
  - BertSentenceEmbeddings      Spark NLP: runs BERT distributed across all cores
                                 (model: sent_bert_base_uncased  768-dim)
  - Spark DataFrame transforms  in-memory distributed NLP pipeline
  - PyArrow                     writes resulting embedding Parquet
                                 (bypasses Hadoop NativeIO Windows write bug,
                                  all heavy processing happens inside Spark JVM)

ALL papers per year are processed (no arbitrary limit).
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BertSentenceEmbeddings
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# ── Configuration ─────────────────────────────────────────────────────────────
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["JAVA_HOME"]   = r"C:\Program Files\Java\jdk-18.0.2.1"

BASE_DIR      = r"c:\Users\Afham Faiyaz Ahmad\Desktop\big data\DriftScope-main"
INPUT_DIR     = os.path.join(BASE_DIR, "data_lake")
OUTPUT_DIR    = os.path.join(BASE_DIR, "embeddings_lake")
TARGET_YEARS  = [2010, 2015, 2020, 2023, 2025]
BATCH_SIZE    = 8        # Spark NLP BERT batch size (conservative for CPU RAM)

OUT_SCHEMA = pa.schema([
    pa.field("id",               pa.string()),
    pa.field("title",            pa.string()),
    pa.field("categories",       pa.string()),
    pa.field("embedding_vector", pa.list_(pa.float32())),
])


def main():
    print("=" * 65)
    print("DriftScope — Stage 2: Spark NLP BERT Embeddings")
    print("=" * 65)

    # ── Start Spark with NLP jars (sparknlp.start) ───────────────────────────
    print("\nStarting Spark NLP session...")
    spark = sparknlp.start(
        spark_nlp_version="5.3.3",
        # gpu=False  — set True if CUDA is available to run BERT on GPU
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark version  : {spark.version}")
    print(f"Spark NLP      : {sparknlp.version()}")
    print(f"Cores in use   : {spark.sparkContext.defaultParallelism}")

    # ── Build Spark NLP pipeline ──────────────────────────────────────────────
    # DocumentAssembler: converts the `abstract` string column into
    # a Spark NLP `Document` annotation that BERT can operate on.
    document_assembler = (
        DocumentAssembler()
        .setInputCol("abstract")
        .setOutputCol("document")
        .setCleanupMode("shrink")
    )

    # BertSentenceEmbeddings: distributed BERT inference across Spark partitions.
    # Downloads sent_bert_base_uncased (~400 MB) on first run, then caches.
    bert_embeddings = (
        BertSentenceEmbeddings
        .pretrained("sent_bert_base_uncased", "en")
        .setInputCols(["document"])
        .setOutputCol("sentence_embedding")
        .setMaxSentenceLength(512)
        .setCaseSensitive(False)
        .setBatchSize(BATCH_SIZE)
    )

    nlp_pipeline = Pipeline(stages=[document_assembler, bert_embeddings])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for yr in TARGET_YEARS:
        print(f"\n{'-'*55}")
        print(f"Year: {yr}")

        in_path  = os.path.join(INPUT_DIR,  f"year={yr}", "part-00000.parquet")
        out_dir  = os.path.join(OUTPUT_DIR, f"year={yr}")
        out_path = os.path.join(out_dir, "part-00000.parquet")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_path):
            print(f"  SKIP — no input parquet at {in_path}")
            continue

        # Read year's parquet (PyArrow, avoid Spark read schema issue)
        pf = pq.ParquetFile(in_path)
        df_pd = (
            pf.read(columns=["id", "title", "abstract", "categories"])
              .to_pandas()
        )
        df_pd = df_pd[df_pd["abstract"].notna() & (df_pd["abstract"].str.strip() != "")]
        print(f"  Papers loaded : {len(df_pd):,}")

        # Convert to Spark DataFrame for NLP pipeline
        spark_df = spark.createDataFrame(df_pd[["id", "title", "abstract", "categories"]])

        # Fit + transform with Spark NLP (BERT runs distributed in Spark JVM)
        print("  Running Spark NLP BERT pipeline...")
        fitted   = nlp_pipeline.fit(spark_df)
        result   = fitted.transform(spark_df)

        # Extract first sentence embedding (768-dim vector) from annotation
        out_df = result.select(
            col("id").cast("string"),
            col("title").cast("string"),
            col("categories").cast("string"),
            col("sentence_embedding.embeddings")[0].alias("embedding_vector"),
        ).filter(col("embedding_vector").isNotNull())

        # Collect to driver, write with PyArrow
        out_pd = out_df.toPandas()
        print(f"  Embeddings generated : {len(out_pd):,}")

        tbl = pa.Table.from_pandas(
            out_pd.assign(
                embedding_vector=out_pd["embedding_vector"].apply(
                    lambda v: list(map(float, v)) if v is not None else None
                )
            ),
            schema=OUT_SCHEMA,
            preserve_index=False,
        )
        pq.write_table(tbl, out_path, compression="snappy")
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved : {out_path}  ({size_mb:.1f} MB)")

    print("\n" + "=" * 65)
    print("Stage 2 complete — embeddings_lake/ ready.")
    print("=" * 65)
    spark.stop()


if __name__ == "__main__":
    main()
