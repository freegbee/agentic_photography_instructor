import os
from mlflow import MlflowClient

def main():
    # FIXME adjust port accoring to config
    print('Trying to start...')

    mlflow_tracking_url=os.environ["MLFLOW_TRACKING_URI"]
    print(f'Trying to create client at {mlflow_tracking_url}...')
    client = MlflowClient(tracking_uri=mlflow_tracking_url)

    # Provide an Experiment description that will appear in the UI
    experiment_description = (
        "Agantic Photography Instructor MLFlow PoC"
    )

    # Provide searchable tags that define characteristics of the Runs that
    # will be in this Experiment
    experiment_tags = {
        "project_name": "api_poc",
        "mlflow.note.content": experiment_description,
    }

    # Create the Experiment, providing a unique name
    produce_apples_experiment = client.create_experiment(
        name="API_Poc_0.2", tags=experiment_tags
    )


if __name__ == "__main__":
    print('Train...')
    main()
