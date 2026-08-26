from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    to_date,
    current_timestamp,
    row_number
)
import argparse
import re
from datetime import datetime
from pyspark.sql.window import Window
from pyspark.sql.functions import when, concat_ws

def get_latest_snapshot(base_path, prefix, process_date):
    """
    Find the latest snapshot file whose date is <= process_date.

    Example:
        customers_2026-08-20.csv
        customers_2026-08-21.csv

    For process_date = 2026-08-22,
    returns customers_2026-08-21.csv.
    """

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

    path = spark._jvm.org.apache.hadoop.fs.Path(base_path)

    fs = path.getFileSystem(hadoop_conf)

    files = fs.listStatus(path)

    process_dt = datetime.strptime(
        process_date,
        "%Y-%m-%d"
    )

    candidates = []

    pattern = re.compile(
        rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv"
    )

    for file_status in files:

        filename = file_status.getPath().getName()

        match = pattern.match(filename)

        if match:

            file_date = datetime.strptime(
                match.group(1),
                "%Y-%m-%d"
            )

            if file_date <= process_dt:
                candidates.append(
                    (file_date, file_status.getPath().toString())
                )

    if not candidates:
        raise FileNotFoundError(
            f"No {prefix} snapshot found "
            f"on or before {process_date}"
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    latest_file = candidates[0][1]

    print(
        f"Latest {prefix} snapshot: "
        f"{latest_file}"
    )

    return latest_file

parser = argparse.ArgumentParser()

parser.add_argument(
    "--process_date",
    required=True
)

args = parser.parse_args()

process_date = args.process_date

print(f"Processing date: {process_date}")

spark = (
    SparkSession.builder
    .appName("EcommerceFullETL")
    .getOrCreate()
)

# ==========================================================
# CONFIGURATION
# ==========================================================

BUCKET = "ecommerce-data-platform"
PROJECT = "ecom-project-506218"

CUSTOMERS_DIR = f"gs://{BUCKET}/raw/customers"
PRODUCTS_DIR = f"gs://{BUCKET}/raw/products"

CUSTOMERS_PATH = get_latest_snapshot(
    CUSTOMERS_DIR,
    "customers",
    process_date
)

PRODUCTS_PATH = get_latest_snapshot(
    PRODUCTS_DIR,
    "products",
    process_date
)

ORDERS_PATH = (
    f"gs://{BUCKET}/"
    f"raw/orders/orders_{process_date}.csv"
)

EVENTS_PATH = f"gs://{BUCKET}/raw/events/*.csv"

# ==========================================================
# READ DATA FROM GCS CSV
# ==========================================================

customers = (
    spark.read
    .option("header","true")
    .option("inferSchema","true")
    .csv(CUSTOMERS_PATH)
)

products = (
    spark.read
    .option("header","true")
    .option("inferSchema","true")
    .csv(PRODUCTS_PATH)
)

orders = (
    spark.read
    .option("header","true")
    .option("inferSchema","true")
    .csv(ORDERS_PATH)
)

events = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(EVENTS_PATH)
)

# Remove duplicate primary keys

customers = customers.dropDuplicates(["customer_id"])
products = products.dropDuplicates(["product_id"])
orders = orders.dropDuplicates(["order_id"])
events = events.dropDuplicates(["event_id"])

# Remove invalid records(where ids are null)

customers = customers.filter(
    col("customer_id").isNotNull()
)

products = products.filter(
    col("product_id").isNotNull()
)

orders = orders.filter(
    col("order_id").isNotNull()
)


# Handle null discount values 
orders = orders.fillna(
    {"discount": 0.0}
)


# Cast data types according to data
orders = (
    orders
    .withColumn(
        "order_date",
        to_date("order_date")
    )
    .withColumn(
        "quantity",
        col("quantity").cast("int")
    )
    .withColumn(
        "price",
        col("price").cast("double")
    )
    .withColumn(
        "discount",
        col("discount").cast("double")
    )
)

# Create Dim tables
dim_customer = (
    customers
    .select(
        "customer_id",
        "name",
        "age",
        "city",
        "signup_date"
    )
)

dim_product = (
    products
    .select(
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "price",
        "cost"
    )
)

# Create Fact table 

fact_orders = (
    orders
    .join(
        dim_customer.select("customer_id"),
        "customer_id",
        "left"
    )
    .join(
        dim_product.select("product_id"),
        "product_id",
        "left"
    )
    .withColumn(
        "gross_amount",
        col("quantity")*col("price")
    )
    .withColumn(
        "net_amount",
        (col("quantity") * col("price")) - col("discount")
    )
    .withColumn(
        "etl_loaded_at",
        current_timestamp()
    )
    .select(
        "order_id",
        "customer_id",
        "product_id",
        "order_date",
        "quantity",
        "price",
        "discount",
        "gross_amount",
        "net_amount",
        "payment_method",
        "etl_loaded_at"
    )
)

# ==========================================================
# DATA QUALITY VALIDATION
# ==========================================================

print("Starting data quality validation...")

# ----------------------------------------------------------
# 1. Detect duplicate order IDs
# ----------------------------------------------------------

window_spec = Window.partitionBy(
    "order_id"
).orderBy(
    col("etl_loaded_at").desc()
)

orders_with_rank = fact_orders.withColumn(
    "duplicate_rank",
    row_number().over(window_spec)
)


# ----------------------------------------------------------
# 2. Define invalid-record conditions
# ----------------------------------------------------------

rejected_with_reason = (
    orders_with_rank
    .withColumn(
        "rejection_reason",
        concat_ws(
            ", ",
            when(
                col("order_id").isNull(),
                lit("NULL_ORDER_ID")
            ),
            when(
                col("customer_id").isNull(),
                lit("NULL_CUSTOMER_ID")
            ),
            when(
                col("product_id").isNull(),
                lit("NULL_PRODUCT_ID")
            ),
            when(
                col("order_date").isNull(),
                lit("NULL_ORDER_DATE")
            ),
            when(
                col("quantity") <= 0,
                lit("INVALID_QUANTITY")
            ),
            when(
                col("price") <= 0,
                lit("INVALID_PRICE")
            ),
            when(
                col("duplicate_rank") > 1,
                lit("DUPLICATE_ORDER_ID")
            )
        )
    )
)

# ----------------------------------------------------------
# 3. Separate valid and rejected records
# ----------------------------------------------------------

rejected_orders = (
    rejected_with_reason
    .filter(col("rejection_reason") != "")
    .drop("duplicate_rank")
)

valid_orders = (
    rejected_with_reason
    .filter(col("rejection_reason") == "")
    .drop("duplicate_rank", "rejection_reason")
)


# ----------------------------------------------------------
# 4. Record counts
# ----------------------------------------------------------

total_count = fact_orders.count()
valid_count = valid_orders.count()
rejected_count = rejected_orders.count()

print("Rejected record details:")

rejected_orders.select(
    "order_id",
    "quantity",
    "price",
    "customer_id",
    "product_id",
    "order_date",
    "rejection_reason"
).show(
    truncate=False
)

print(f"Total records: {total_count}")
print(f"Valid records: {valid_count}")
print(f"Rejected records: {rejected_count}")

# ==========================================================
# WRITE REJECTED RECORDS
# ==========================================================

REJECTED_PATH = (
    f"gs://{BUCKET}/"
    f"rejected/orders/process_date={process_date}"
)

if rejected_count > 0:

    (
        rejected_orders.write
        .mode("overwrite")
        .option("header", "true")
        .csv(REJECTED_PATH)
    )

    print(
        f"Rejected records written to: "
        f"{REJECTED_PATH}"
    )

else:

    print("No rejected records found.")

# Write to BigQuery Tables 
(
    dim_customer.write
    .format("bigquery")
    .option(
        "table",
        f"{PROJECT}.curated.dim_customer"
    )
    .mode("overwrite")
    .save()    
)

(
    dim_product.write
    .format("bigquery")
    .option(
        "table",
        f"{PROJECT}.curated.dim_product"
    )
    .mode("overwrite")
    .save()
)

FACT_STAGING_TABLE = (
    f"{PROJECT}.staging.fact_orders_staging_{process_date.replace('-', '')}"
)

(
    valid_orders.write
    .format("bigquery")
    .option(
        "table",
        FACT_STAGING_TABLE
    )
    .mode("overwrite")
    .save()
)

print(
    f"Fact orders written to staging table: "
    f"{FACT_STAGING_TABLE}"
)

if valid_count == 0:

    raise RuntimeError(
        f"No valid orders found for process_date={process_date}. "
        "Stopping pipeline."
    )

print("Full ETL completed successfully.")

spark.stop()