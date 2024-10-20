import pandas as pd
from ucimlrepo import fetch_ucirepo
import sys

class DataIngest():
    @staticmethod
    def data_ingest():
        # 162 for forest fires
        dataset_ = fetch_ucirepo(id=162)
        X = dataset_.data.features
        y = dataset_.data.targets
        return pd.concat([X, y], axis = 1)
    
if __name__ == '__main__':
    output_file = sys.argv[1]
    data = DataIngest.data_ingest()
    data.to_csv(output_file, index=False)
