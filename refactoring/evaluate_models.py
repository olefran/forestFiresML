import sys
import yaml
import joblib
import mlflow

class EvaluateModels():
    """
    A class representing EvaluateModels object.

    Attributes:
        params (Dict): Dictionary of params from yaml params file.
        model_dir (str): Path for the models
    """
    def __init__(self, params_path):
        """
        Initializes a EvaluateModels object, reading data from the params yamkl file path.

        Parameters:
            params_path (str): The path of params yaml file.
        """
        with open(params_path, 'r') as ymlfile:
            self.params = yaml.safe_load(ymlfile)

        self.model_dir = self.params['data']['models']
    
    def evaluate_models(self):
        """
        Perform evaluating models, final defining champion model.

        Parameters:

        Returns:

        """
        # Fetch all runs from the experiment
        experiment_id = mlflow.get_experiment_by_name("MLForestFires").experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id])

        # Find the run with the lowest RMSE
        best_run = runs.sort_values("metrics.rmse", ascending=True).iloc[0]

        print(f"Best run is the model {best_run['tags.mlflow.runName']} with RMSE: {best_run['metrics.rmse']} with ID:{best_run['run_id']}")

        model_path = f"{self.model_dir}/{best_run['tags.mlflow.runName']}.pkl"
        final_model = joblib.load(model_path)
        joblib.dump(final_model, f"{self.model_dir}/final_model.pkl")

        return

if __name__ == '__main__':
    params_path = sys.argv[1]

    evm = EvaluateModels(params_path)
    evm.evaluate_models()
