# Databricks notebook source
import mlflow

def find_latest_experiment() -> mlflow.entities.experiment.Experiment:
  """
  Finds the latest experiment run by the current user
    
  Parameters:
  - None
  
  Returns:
  - mlflow.entities.experiment.Experiment: The most recent experiment performed by the user.
  """
  # Initialize the MLflow client
  client = mlflow.tracking.MlflowClient()

  # Fetch all available experiments
  experiments = client.search_experiments()

  # Sort the experiments by their last update time in descending order
  sorted_experiments = sorted(experiments, key=lambda x: x.last_update_time, reverse=True)

  # Retrieve the most recently updated experiment
  latest_experiment = sorted_experiments[0]

  # Output the name of the latest experiment
  print(f"The most recently updated experiment is named '{latest_experiment.name}'.")
  return sorted_experiments[0]
  

# COMMAND ----------

import mlflow

def find_best_run_id(experiment: mlflow.entities.experiment.Experiment, metric: str) -> str:
  """
  finds the run_id from the given experiment according to the provided metric
    
    Parameters:
    experiment - mlflow Experiment the user wants to look up to find the best run
    metric - how the best run will be decided
    
    Returns:
    - string: The most recent experiment performed by the user.
  """
  # Initialize the MLflow client
  client = mlflow.tracking.MlflowClient()

  # Initialize the Databricks utilities to programmatically fetch the username
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

  # Retrieve the name of the latest experiment; assumed to have been set in earlier steps
  experiment_name = experiment.name

  # Fetch the experiment details using its name
  experiment_details = client.get_experiment_by_name(experiment_name)

  # Search for runs within the experiment and sort them by validation F1 score in descending order
  sorted_runs = mlflow.search_runs(experiment_details.experiment_id).sort_values(metric, ascending=False)

  # Get the run ID of the best model based on the highest validation F1 score
  best_run_id = sorted_runs.loc[0, "run_id"]
  print("The best run id is: ", best_run_id)
  # Note: The variable `best_run_id` now contains the run ID of the best model in the specified experiment
  return best_run_id
  


# COMMAND ----------

import mlflow

def register_run_as_model(registry_model_name: str, run_id: str) -> None:
  """
  finds the run_id from the given experiment according to the provided metric
    
    Parameters:
    experiment - mlflow Experiment the user wants to look up to find the best run
    metric - how the best run will be decided
    
    Returns:
    - string: The most recent experiment performed by the user.
  """
  # Initialize the model's URI using the best run ID obtained from previous steps
  model_uri = f"runs:/{run_id}/model"

  # Register the model in MLflow's model registry under the specified name
  try:
      model_details = mlflow.register_model(model_uri=model_uri, name=registry_model_name)
      print(f"Successfully registered model '{registry_model_name}' with URI '{model_uri}'.")
  except mlflow.exceptions.MlflowException as e:
      print(f"Failed to register model '{registry_model_name}': {str(e)}")

  print("Registered model details are: \n", model_details)
  return model_details

# COMMAND ----------

import mlflow
new_description = "This model predicts the show up rate for successful applications."

def update_registered_model_description(model_details: mlflow.entities.model_registry.model_version.ModelVersion, new_description: str):
  """
  Updates the description of an already created model for the entire model

  Parameters:
  - model_details: The `model_details` variable is assumed to contain details about the registered model and its version
  - new_description: The new description to update in the model registry

  Returns: None
  """
  # Initialize the MLflow client
  client = mlflow.tracking.MlflowClient()
  # Update the metadata of an already registered model
  try:
      client.update_registered_model(
          name=model_details.name,
          description=new_description
      )
      print(f"Successfully updated the description for the registered model '{model_details.name}'.")
  except mlflow.exceptions.MlflowException as e:
      print(f"Failed to update the registered model '{model_details.name}': {str(e)}")

new_description = "This is a scikit-learn random forest based model."
def update_registered_model_description(model_details: mlflow.entities.model_registry.model_version.ModelVersion, new_description: str):
  """
  Updates the description of an already created model for a specific version of the model

  Parameters:
  - model_details: The `model_details` variable is assumed to contain details about the registered model and its version
  - new_description: The new description to update in the model registry

  Returns: None
  """
  # Initialize the MLflow client
  client = mlflow.tracking.MlflowClient()
  # Update the metadata for a specific version of the model
  try:
      client.update_model_version(
          name=model_details.name,
          version=model_details.version,
          description=new_description
      )
      print(f"Successfully updated the description for version {model_details.version} of the model '{model_details.name}'.")
  except mlflow.exceptions.MlflowException as e:
      print(f"Failed to update version {model_details.version} of the model '{model_details.name}': {str(e)}")

  

# COMMAND ----------

import mlflow

def update_model_stage(model_details: mlflow.entities.model_registry.model_version.ModelVersion, new_stage: str='Staging') -> None:
  """
  finds the run_id from the given experiment according to the provided metric
    
    Parameters:
    - model_details: The `model_details` variable is assumed to contain details about the registered model and its version
    - new_stage: the stage the model is supposed to be moved to
    
    Returns:
    - None
  """
  # Initialize the MLflow client
  client = mlflow.tracking.MlflowClient()

  # Transition the model version to the 'Staging' stage in the model registry
  try:
      client.transition_model_version_stage(
          name=model_details.name,
          version=model_details.version,
          stage=new_stage,
          archive_existing_versions=True  # Archives any existing versions in the 'Staging' stage
      )
      print(f"Successfully transitioned version {model_details.version} of the model '{model_details.name}' to '{new_stage}'.")
  except mlflow.exceptions.MlflowException as e:
      print(f"Failed to transition version {model_details.version} of the model '{model_details.name}' to '{new_stage}': {str(e)}")
