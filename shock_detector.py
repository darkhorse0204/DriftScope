from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import os

def main():
    print("Initializing DriftScope Semantic Shock Detector...")
    spark = SparkSession.builder \
        .appName("DriftScope_ShockDetector") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    base_dir = "c:/Users/Afham Faiyaz Ahmad/Desktop/big data/DriftScope-main"
    stream_input_dir  = f"{base_dir}/stream_input"
    stream_output_dir = f"{base_dir}/stream_output"
    checkpoint_dir    = f"{base_dir}/checkpoints"

    # Create directories if they don't exist
    os.makedirs(stream_input_dir,  exist_ok=True)
    os.makedirs(stream_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir,    exist_ok=True)

    # Schema matching the drift_results.csv format
    schema = StructType([
        StructField("Topic",                   StringType(), True),
        StructField("Year",                    StringType(), True),
        StructField("Semantic_Drift_Velocity", FloatType(),  True),
    ])

    print(f"\nWatching for CSV files in: {stream_input_dir}")
    print("Drop a CSV (with columns: Topic, Year, Semantic_Drift_Velocity) to simulate live data.")
    print("Shock events (V_sd > 0.10) will be written to: stream_output/")
    print("Press Ctrl+C to stop.\n")

    # Read stream from the watch folder
    stream_df = spark.readStream \
        .schema(schema) \
        .csv(stream_input_dir)

    # Filter: only pass through rows where velocity exceeds the shock threshold
    shock_df = stream_df \
        .filter(col("Semantic_Drift_Velocity") > 0.10) \
        .withColumnRenamed("Semantic_Drift_Velocity", "V_sd")

    # Write detected shocks to CSV output
    query = shock_df.writeStream \
        .outputMode("append") \
        .format("csv") \
        .option("header", "true") \
        .option("path", stream_output_dir) \
        .option("checkpointLocation", checkpoint_dir) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()
