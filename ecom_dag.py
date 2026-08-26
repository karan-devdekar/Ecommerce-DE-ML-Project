from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor

from datetime import datetime, timedelta
import time


PROJECT_ID = "ecom-project-506218"
REGION = "asia-south1"
SPARK_CLUSTER = "ecom-dev-cluster"
BUCKET = "ecommerce-data-platform"
PYSPARK_FILE = f"gs://{BUCKET}/code/full_etl.py"

MODEL_ID = "8073533562852212736"
MODEL_NAME = f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}"


def run_vertex_batch_prediction(**context):

    import time
    from google.cloud import aiplatform_v1

    logical_date = context["logical_date"]

    process_date = (
        logical_date - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    client_options = {
        "api_endpoint": f"{REGION}-aiplatform.googleapis.com"
    }

    client = aiplatform_v1.JobServiceClient(
        client_options=client_options
    )

    INPUT_TABLE = (
        f"bq://{PROJECT_ID}.analytics.customer_segmentation_input"
    )

    OUTPUT_TABLE = (
        f"bq://{PROJECT_ID}.analytics.customer_segmentation_predictions"
    )

    input_config = {
        "instances_format": "bigquery",
        "bigquery_source": {
            "input_uri": INPUT_TABLE
        }
    }

    instance_config = {
        "instance_type": "array",
        "included_fields": [
            "age",
            "total_orders",
            "total_units",
            "total_spend",
            "average_order_value",
            "customer_lifetime_days",
            "recency_days",
        ],
    }

    output_config = {
        "predictions_format": "bigquery",
        "bigquery_destination": {
            "output_uri": OUTPUT_TABLE
        }
    }

    batch_prediction_job = {
        "display_name": (
            f"customer-segmentation-"
            f"{process_date.replace('-', '')}"
        ),

        "model": MODEL_NAME,

        "input_config": input_config,

        "instance_config": instance_config,

        "output_config": output_config,

        "dedicated_resources": {
            "machine_spec": {
                "machine_type": "n1-standard-2"
            },
            "starting_replica_count": 1,
            "max_replica_count": 1,
        },
    }

    parent = (
        f"projects/{PROJECT_ID}/locations/{REGION}"
    )

    job = client.create_batch_prediction_job(
        parent=parent,
        batch_prediction_job=batch_prediction_job,
    )

    print(
        f"Created Vertex AI batch prediction job: "
        f"{job.name}"
    )

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:

        current_job = client.get_batch_prediction_job(
            name=job.name
        )

        state_name = current_job.state.name

        print(
            f"Vertex AI job state: {state_name}"
        )

        if state_name in terminal_states:

            if state_name != "JOB_STATE_SUCCEEDED":

                error = getattr(
                    current_job,
                    "error",
                    None
                )

                raise RuntimeError(
                    f"Vertex AI batch prediction failed: "
                    f"{state_name}; error={error}"
                )

            failed_count = (
                current_job.completion_stats.failed_count
            )

            print(
                "Vertex AI batch prediction "
                f"succeeded. Failed predictions: "
                f"{failed_count}"
            )

            # IMPORTANT:
            # Stop polling once Vertex succeeds.
            break

        time.sleep(30)

    return job.name


with DAG(
    dag_id="ecommerce_pipeline",
    start_date=datetime(2026, 8, 21),
    schedule="@daily",
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["ecommerce", "spark", "bigquery", "vertex-ai"],
) as dag:

    start = EmptyOperator(task_id="start")

    check_orders_file = GCSObjectExistenceSensor(
        task_id="check_orders_file",
        bucket=BUCKET,
        object="raw/orders/orders_{{ (logical_date - macros.timedelta(days=1)).strftime('%Y-%m-%d') }}.csv",
        deferrable=True,
        poke_interval=60,
        timeout=60 * 60,
    )

    pyspark_job = {
        "reference": {"project_id": PROJECT_ID},
        "placement": {"cluster_name": SPARK_CLUSTER},
        "pyspark_job": {
            "main_python_file_uri": PYSPARK_FILE,
            "args": ["--process_date", "{{ (logical_date - macros.timedelta(days=1)).strftime('%Y-%m-%d') }}"],
        },
    }

    run_spark_etl = DataprocSubmitJobOperator(
        task_id="run_spark_etl",
        project_id=PROJECT_ID,
        region=REGION,
        job=pyspark_job,
        deferrable=True,
    )

    merge_fact_orders = BigQueryInsertJobOperator(
        task_id="merge_fact_orders",
        project_id=PROJECT_ID,
        location=REGION,
        configuration={
            "query": {
                "query": f"""
                    MERGE `{PROJECT_ID}.curated.fact_orders` AS target
                    USING (
                        SELECT *
                        FROM `{PROJECT_ID}.staging.fact_orders_staging_{{{{ (logical_date - macros.timedelta(days=1)).strftime('%Y%m%d') }}}}`
                    ) AS source
                    ON target.order_id = source.order_id
                    AND target.customer_id = source.customer_id

                    WHEN MATCHED THEN
                        UPDATE SET
                            customer_id = source.customer_id,
                            product_id = source.product_id,
                            order_date = source.order_date,
                            quantity = source.quantity,
                            price = source.price,
                            discount = source.discount,
                            gross_amount = source.gross_amount,
                            net_amount = source.net_amount,
                            payment_method = source.payment_method,
                            etl_loaded_at = source.etl_loaded_at

                    WHEN NOT MATCHED THEN
                        INSERT (
                            order_id,
                            customer_id,
                            product_id,
                            order_date,
                            quantity,
                            price,
                            discount,
                            gross_amount,
                            net_amount,
                            payment_method,
                            etl_loaded_at
                        )
                        VALUES (
                            source.order_id,
                            source.customer_id,
                            source.product_id,
                            source.order_date,
                            source.quantity,
                            source.price,
                            source.discount,
                            source.gross_amount,
                            source.net_amount,
                            source.payment_method,
                            source.etl_loaded_at
                        )
                """,
                "useLegacySql": False,
            }
        },
        deferrable=True,
    )

    daily_sales = BigQueryInsertJobOperator(
        task_id="daily_sales",
        project_id=PROJECT_ID,
        location=REGION,
        configuration={
            "query": {
                "query": f"""
                    CREATE OR REPLACE TABLE
                    `{PROJECT_ID}.analytics.daily_sales` AS
                    SELECT
                        order_date,
                        COUNT(DISTINCT order_id) AS total_orders,
                        SUM(quantity) AS total_units,
                        ROUND(SUM(gross_amount), 2) AS gross_sales,
                        ROUND(SUM(discount), 2) AS total_discount,
                        ROUND(SUM(net_amount), 2) AS net_sales
                    FROM `{PROJECT_ID}.curated.fact_orders`
                    GROUP BY order_date
                    ORDER BY order_date
                """,
                "useLegacySql": False,
            }
        },
        deferrable=True,
    )

    product_performance = BigQueryInsertJobOperator(
        task_id="product_performance",
        project_id=PROJECT_ID,
        location=REGION,
        configuration={
            "query": {
                "query": f"""
                    CREATE OR REPLACE TABLE
                    `{PROJECT_ID}.analytics.product_performance` AS
                    SELECT
                        p.product_id,
                        p.product_name,
                        p.category,
                        p.subcategory,
                        COUNT(DISTINCT f.order_id) AS total_orders,
                        SUM(f.quantity) AS units_sold,
                        ROUND(SUM(f.gross_amount), 2) AS gross_sales,
                        ROUND(SUM(f.discount), 2) AS discounts,
                        ROUND(SUM(f.net_amount), 2) AS net_sales,
                        ROUND(
                            SUM(f.net_amount - (f.quantity * p.cost)),
                            2
                        ) AS profit
                    FROM `{PROJECT_ID}.curated.fact_orders` f
                    JOIN `{PROJECT_ID}.curated.dim_product` p
                      ON f.product_id = p.product_id
                    GROUP BY
                        p.product_id,
                        p.product_name,
                        p.category,
                        p.subcategory
                """,
                "useLegacySql": False,
            }
        },
        deferrable=True,
    )

    customer_360 = BigQueryInsertJobOperator(
        task_id="customer_360",
        project_id=PROJECT_ID,
        location=REGION,
        configuration={
            "query": {
                "query": f"""
                    CREATE OR REPLACE TABLE
                    `{PROJECT_ID}.analytics.customer_360` AS
                    SELECT
                        c.customer_id,
                        c.name,
                        c.age,
                        c.city,
                        c.signup_date,
                        COUNT(DISTINCT f.order_id) AS total_orders,
                        COALESCE(SUM(f.quantity), 0) AS total_units,
                        ROUND(COALESCE(SUM(f.net_amount), 0), 2) AS total_spend,
                        ROUND(COALESCE(AVG(f.net_amount), 0), 2) AS average_order_value,
                        MIN(f.order_date) AS first_order_date,
                        MAX(f.order_date) AS last_order_date
                    FROM `{PROJECT_ID}.curated.dim_customer` c
                    LEFT JOIN `{PROJECT_ID}.curated.fact_orders` f
                      ON c.customer_id = f.customer_id
                    GROUP BY
                        c.customer_id,
                        c.name,
                        c.age,
                        c.city,
                        c.signup_date
                """,
                "useLegacySql": False,
            }
        },
        deferrable=True,
    )

    vertex_batch_prediction = PythonOperator(
        task_id="vertex_batch_prediction",
        python_callable=run_vertex_batch_prediction,
    )

    create_customer_segments = BigQueryInsertJobOperator(
        task_id="create_customer_segments",
        project_id=PROJECT_ID,
        location=REGION,
        configuration={
            "query": {
                "query": f"""
                    CREATE OR REPLACE TABLE
                    `{PROJECT_ID}.analytics.customer_segments` AS
                    SELECT
                        p.customer_id,
                        p.prediction AS cluster_id,
                        CASE
                            WHEN p.prediction = 1 THEN 'High-Value Customers'
                            WHEN p.prediction = 0 THEN 'Regular Customers'
                            WHEN p.prediction = 2 THEN 'Emerging Customers'
                            ELSE 'Unknown'
                        END AS customer_segment,
                        f.total_orders,
                        f.total_spend,
                        f.average_order_value,
                        f.customer_lifetime_days,
                        f.recency_days
                    FROM
                        `{PROJECT_ID}.analytics.customer_segmentation_predictions` p
                    JOIN
                        `{PROJECT_ID}.analytics.customer_ml_features` f
                      ON CAST(p.customer_id AS INT64) = f.customer_id
                """,
                "useLegacySql": False,
            }
        },
        deferrable=True,
    )

    end = EmptyOperator(task_id="end")

    start >> check_orders_file >> run_spark_etl >> merge_fact_orders

    merge_fact_orders >> daily_sales >> product_performance >> customer_360

    customer_360 >> vertex_batch_prediction
    vertex_batch_prediction >> create_customer_segments >> end