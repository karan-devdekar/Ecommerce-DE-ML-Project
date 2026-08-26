import os
import joblib
import pandas as pd

from google.cloud import bigquery
from google.cloud import storage

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


PROJECT_ID = "ecom-project-506218"
DATASET = "analytics"
TABLE = "customer_ml_features"

FEATURES = [
    "age",
    "total_orders",
    "total_units",
    "total_spend",
    "average_order_value",
    "customer_lifetime_days",
    "recency_days",
]

NUM_CLUSTERS = 3


def main():

    print("========================================")
    print("Starting customer segmentation training")
    print("========================================")

    # -----------------------------------------
    # 1. Read data from BigQuery
    # -----------------------------------------

    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT
            customer_id,
            {", ".join(FEATURES)}
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
    """

    print("Reading customer features from BigQuery...")

    df = client.query(query).to_dataframe()

    print(f"Customers loaded: {len(df)}")

    if df.empty:
        raise RuntimeError("No customer records found.")

    # -----------------------------------------
    # 2. Prepare features
    # -----------------------------------------

    X = df[FEATURES].copy()

    X = X.fillna(0)

    print("\nFeatures used for training:")
    for feature in FEATURES:
        print(f" - {feature}")

    # -----------------------------------------
    # 3. Create ML pipeline
    # -----------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "kmeans",
                KMeans(
                    n_clusters=NUM_CLUSTERS,
                    random_state=42,
                    n_init=10
                )
            )
        ]
    )

    # -----------------------------------------
    # 4. Train K-Means
    # -----------------------------------------

    print("\nTraining K-Means model...")

    pipeline.fit(X)

    # -----------------------------------------
    # 5. Generate cluster assignments
    # -----------------------------------------

    df["cluster_id"] = pipeline.predict(X)

    print("\nCluster distribution:")

    print(
        df["cluster_id"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------
    # 6. Generate cluster summary
    # -----------------------------------------

    cluster_summary = (
        df
        .groupby("cluster_id")[FEATURES]
        .mean()
        .round(2)
    )

    print("\nCluster summary:")
    print(cluster_summary)

    # -----------------------------------------
    # 7. Calculate model quality
    # -----------------------------------------

    kmeans = pipeline.named_steps["kmeans"]

    print("\nModel information:")
    print(f"Number of clusters: {NUM_CLUSTERS}")
    print(f"Inertia: {kmeans.inertia_}")

    # -----------------------------------------
    # 8. Get AIP_MODEL_DIR
    # -----------------------------------------

    model_dir = os.environ.get("AIP_MODEL_DIR")

    if not model_dir:
        raise RuntimeError(
            "AIP_MODEL_DIR environment variable is not set."
        )

    print(f"\nAIP_MODEL_DIR: {model_dir}")

    # -----------------------------------------
    # 9. Save model locally
    # -----------------------------------------

    local_model_path = "/tmp/model.joblib"

    joblib.dump(
        pipeline,
        local_model_path
    )

    print(
        f"Model saved locally: {local_model_path}"
    )

    # -----------------------------------------
    # 10. Upload model to Cloud Storage
    # -----------------------------------------

    storage_client = storage.Client()

    model_directory = model_dir.replace(
        "gs://",
        ""
    )

    bucket_name, blob_prefix = model_directory.split(
        "/",
        1
    )

    bucket = storage_client.bucket(
        bucket_name
    )

    blob = bucket.blob(
        f"{blob_prefix.rstrip('/')}/model.joblib"
    )

    blob.upload_from_filename(
        local_model_path
    )

    print(
        f"Model uploaded to: gs://{bucket_name}/{blob.name}"
    )

    print("\n========================================")
    print("Training completed successfully")
    print("========================================")


if __name__ == "__main__":
    main()