import numpy as np
import pandas as pd
import sys

class CleanData():
    """
    A class representing CleanData object.

    Attributes:
        data_path (str): The path od the data ingested.
    """
    def __init__(self, data_path):
        """
        Initializes a CleanData object, reading data from the csv file path.

        Parameters:
            data_path (str): The path od the data ingested.
        """
        self.data_ = pd.read_csv(data_path)

    def clean_data(self):
        """
        Perform data cleaning process.

        Parameters:

        Returns:
            DataFrame: Data cleaned as DataFrame.
        """
        data = self.data_.dropna()

        print(f"\nNúmero de filas duplicadas: {data.duplicated().sum()}")
        data = data.drop_duplicates()
        print(f"Número de filas después de eliminar duplicados: {data.shape[0]}")

        Q1 = data['area'].quantile(0.25)
        Q3 = data['area'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data_filtered = data[(data['area'] >= lower_bound) & (data['area'] <= upper_bound)]
        print(f"Cantidad de datos después de eliminar outliers: {data_filtered.shape[0]}")

        data_filtered['log_area'] = np.log1p(data_filtered['area'])

        return data_filtered
    
if __name__ == '__main__':
    data_path = sys.argv[1]
    output_clean_data = sys.argv[2]

    cld = CleanData(data_path)
    df_clean_data = cld.clean_data()

    df_clean_data.to_csv(output_clean_data, index=False)