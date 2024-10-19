import forestFiresML.feractoringClass as m

def main():

    print("Initializing Data Download on ./data")
    data_ = m.DataImport.data_fetch()

    print("Displaying Histograms")
    m.DataExplorer.plot_histograms(data_)

    print("Preparing data...")

    m.DataExplorer.data_exp_n_prep(data_)

    print("Information about data:")

    m.DataExplorer.explore_data(data_)

    print("Processing data...")

    m.FF_model.data_processing(data_)

    print("Model creation and test run:")

    m.FF_model.model_creation(data_)

    print("Ml_traking requires a MLServer instance running on http://localhost:5000")

    m.FF_model.mlflow_tracking(data_)

    print("Best model with: ")

    m.FF_model.best_model_run(data_)

if __name__ == "__main__":
    main()
