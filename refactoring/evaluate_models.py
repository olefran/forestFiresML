import sys
import yaml
import joblib
import mlflow

class EvaluateModels():
    def __init__(self, params_path):
        with open("params.yaml", 'r') as ymlfile:
            self.params = yaml.safe_load(ymlfile)

        self.model_dir = self.params['data']['models']
    
    def evaluate_models(self):
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
