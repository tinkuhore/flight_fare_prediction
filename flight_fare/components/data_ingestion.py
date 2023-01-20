from flight_fare import utility
from flight_fare.entity import config_entity, artifact_entity
from flight_fare.logger import logging
from flight_fare.exception import FlightFareException
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:
    
    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig) -> None:
        try:
            logging.info(f"\n\n{'>'*20} Data Ingestion {'<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise FlightFareException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data from MongoDB as pandas Dataframe")
            df:pd.DataFrame = utility.get_collection_as_df(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Saving data in feature store ...")
            # drop na values
            df.dropna(inplace=True)

            logging.info(f"Create feature store folder if not available.")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info(f"Save DF to feature store folder.")
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info(f"Split dataset to create Test and Train data")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42)

            logging.info(f"Create dataset folder if not available.")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info(f"Save Train and Test DF to dataset folder.")
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info(f"Prepearing artifact of DataIngestion")
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                frature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion Artifact : {data_ingestion_artifact}")
            return data_ingestion_artifact            
        except Exception as e:
            raise FlightFareException(error_msg=e, error_detail=sys)
