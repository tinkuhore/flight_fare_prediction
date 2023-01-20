import pandas as pd
import numpy as np
from flight_fare.logger import logging
from flight_fare.exception import FlightFareException
from flight_fare.config import mongo_client
import os, sys
import yaml
import dill


def get_collection_as_df(database_name: str, collection_name: str)-> pd.DataFrame:
    """
    Description: Convert the data collected from database into pandas DataFrame
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from Database: {database_name} and Collection : {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Columns found are : {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Droping column : _id")
            df = df.drop("_id", axis=1)
        logging.info(f"Rows and Colmns in df : {df.shape} ")
        return df
    except Exception as e:
        raise FlightFareException(e, sys)

def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise FlightFareException(e, sys)

def save_object(file_path:str, obj:object):
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise FlightFareException(e, sys) from e

def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise FlightFareException(e, sys) from e