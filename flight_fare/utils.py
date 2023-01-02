import pickle
from flight_fare.logger import logging
from flight_fare.exception import FlightFareException
from flight_fare.credentials import *
from typing import *
import sys, os
import pandas as pd
from datetime import datetime

def transform(data: List) ->List:
    """
    This function transfor the collected informations provided by the user into a pandas.DataFrame 
    which contains all necessary features required by the Model for prediction.
    
    returns: pandas.DataFrame 
    """    

    def source_encoder(source: str) ->List:
        try:
            logging.info(f"Source Location : {source}")
            logging.info(f"Applying OneHotEncode on Source")
            
            source_list = columns[8:13]
            logging.info(f"Before Encoding : {source_list}")
            
            for i in range(len(source_list)):
                if source in source_list[i]:
                    source_list[i] = 1
                else:
                    source_list[i] = 0
            
            logging.info(f"After Encoding: {source_list}")
            return source_list
        except Exception as e:
            FlightFareException(e,sys)

    def destination_encoder(destination: str) ->List:
        try:
            logging.info(f"Destination Location : {destination}")
            logging.info(f"Applying OneHotEncode on Destination")
            
            destination_list = columns[13:19]
            logging.info(f"Before Encoding : {destination_list}")
            
            for i in range(len(destination_list)):
                if destination in destination_list[i]:
                    destination_list[i] = 1
                else:
                    destination_list[i] = 0
            
            logging.info(f"After Encoding: {destination_list}")
            return destination_list
        except Exception as e:
            FlightFareException(e,sys)

    def airline_encoder(airline: str) ->List:
        try:
            logging.info(f"Airline Location : {airline}")
            logging.info(f"Applying OneHotEncode on Airline")
            
            airline_list = columns[19:]
            logging.info(f"Before Encoding : {airline_list}")
            
            for i in range(len(airline_list)):
                if airline in airline_list[i]:
                    airline_list[i] = 1
                else:
                    airline_list[i] = 0
            
            logging.info(f"After Encoding: {airline_list}")
            return airline_list
        except Exception as e:
            FlightFareException(e,sys)

    try:
        logging.info(f">>>> Transforming collected data <<<<")
        # data_trf = ['Duration', 'Total_Stops', 'Day', 'Month', 'Dep_hr', 'Dep_min', 'Arrival_hr', 'Arrival_min']
        # data = ['2023-01-11T12:34', '2023-01-11T03:05', 'Delhi', 'Cochin', '0', 'Multiple carriers']            

        logging.info("STEP-1: Transforming Date & Time")
        # datetime type casting 
        dep = datetime.strptime(data[0], "%Y-%m-%dT%H:%M")
        arrv = datetime.strptime(data[1], "%Y-%m-%dT%H:%M")
        
        # calculate duration of flight in minutes
        duration = int((arrv.timestamp() - dep.timestamp())/60)

        # adding first 8 features
        data_trf = [duration, int(data[4]), dep.day, dep.month, dep.hour, dep.minute, arrv.hour, arrv.minute]

        logging.info("STEP-2: Encode source, destination and Airline")
        # OneHotEncode on source, destination and Airline
        data_trf = data_trf + source_encoder(data[2]) + destination_encoder(data[3]) + airline_encoder(data[-1])
        logging.info(f"Transformed Data : {data_trf}")
        
        
        return data_trf
    except Exception as e:
        FlightFareException(e, sys)


class Validate:...

def predict_fare(df: Union[pd.DataFrame, List]) -> float:
    try:
        # load model
        logging.info(f">>>> Predicting Flight Fare  <<<<")
        logging.info(f"Loading Model from .pkl file")        
        gen_model = pickle.load(open('pkl_files/gen_pred_model.pkl', 'rb'))

        predicted_fare = gen_model.predict(df)
        logging.info(f"Predicted Flight Fare = {predicted_fare}")
        return round(predicted_fare[0],2)
    except Exception as e:
        FlightFareException(e, sys)