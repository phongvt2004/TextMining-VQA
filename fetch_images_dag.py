from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.papermill.operators.papermill import PapermillOperator
from datetime import datetime, timedelta
import os

# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'daily_image_fetch',
    default_args=default_args,
    description='Fetch images daily using Google Custom Search API',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 15),
    catchup=False,
) as dag:

    # Task to execute the Jupyter notebook
    run_notebook = PapermillOperator(
        task_id='run_image_request_notebook',
        input_nb='D:/Github/TextMining-VQA/request_image.ipynb',
        output_nb='D:/Github/TextMining-VQA/request_image_output.ipynb',
        parameters={},  
    )

run_notebook
