import pickle
import sys, os
from flight_fare.logger import logging
from flight_fare.exception import FlightFareException

try:
       logging.info(f"Loading all latest models...")
       list_dir = os.listdir("saved_models")
       latest_dir_num = max(list_dir)
       latest_models_dir = os.path.join("saved_models",f"{latest_dir_num}")
       
       gen_model = pickle.load(open(os.path.join(latest_models_dir,'gen_pred_model.pkl'), 'rb'))
       AirAsia = pickle.load(open(os.path.join(latest_models_dir,"AirAsia.pkl"), "rb"))
       IndiGo = pickle.load(open(os.path.join(latest_models_dir,"AirAsia.pkl"), "rb"))
       AirIndia = pickle.load(open(os.path.join(latest_models_dir,"AirIndia.pkl"), "rb"))
       JetAirways = pickle.load(open(os.path.join(latest_models_dir,"JetAirways.pkl"), "rb"))
       SpiceJet = pickle.load(open(os.path.join(latest_models_dir,"SpiceJet.pkl"), "rb"))
       Multiplecarriers = pickle.load(open(os.path.join(latest_models_dir,"Multiplecarriers.pkl"), "rb"))
       GoAir = pickle.load(open(os.path.join(latest_models_dir,"GoAir.pkl"), "rb"))
       Vistara = pickle.load(open(os.path.join(latest_models_dir,"Vistara.pkl"), "rb"))
       logging.info("Loading Complete.")
except Exception as e:
       FlightFareException(e, sys)


columns = ['Duration', 'Total_Stops', 'Day',
       'Month', 'Dep_hr', 'Dep_min', 'Arrival_hr', 'Arrival_min',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata',
       'Source_Mumbai', 'Destination_Banglore', 'Destination_Cochin',
       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi', 'Airline_Air Asia', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Jet Airways Business', 'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy']


airline_model_dict = {
    "AirAsia":AirAsia,
    "IndiGo": IndiGo,
    "AirIndia":AirIndia,
    "JetAirways":JetAirways,
    "SpiceJet":SpiceJet,
    "Multiplecarriers": Multiplecarriers,
    "GoAir":GoAir,
    "Vistara":Vistara
}


avl_airlines = {
       'Banglore': ['IndiGo',  'JetAirways',  'AirIndia',  'Vistara',  'AirAsia',  'SpiceJet',  'GoAir'],
       'Kolkata': ['AirIndia',  'IndiGo',  'SpiceJet',  'JetAirways',  'Vistara',  'GoAir',  'AirAsia'],
       'Delhi': ['JetAirways',  'Multiplecarriers',  'AirIndia',  'SpiceJet',  'GoAir',  'IndiGo',  'Vistara',  'AirAsia'],
       'Chennai': ['AirIndia', 'Vistara', 'IndiGo', 'SpiceJet'], 
       'Mumbai': ['Vistara', 'AirIndia', 'JetAirways', 'IndiGo', 'SpiceJet']
       }