from flight_fare.entity import artifact_entity, config_entity
from flight_fare.exception import FlightFareException
from flight_fare.logger import logging
from flight_fare.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
import os, sys
import pandas as pd
le = LabelEncoder()


class DataTransformation:
    def __init__(self, 
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact, 
                data_transformation_config=config_entity.DataTransformationConfig) -> None:
        try:
            logging.info(f"\n\n{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise FlightFareException(e, sys)

    def transform_numerical_features(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            # type casting of Date_of_journey column
            df.Date_of_Journey = pd.to_datetime(df.Date_of_Journey, dayfirst=True)

            # creating Day and Month columns that contain integer values
            df["Day"] = df.Date_of_Journey.dt.day
            df["Month"] = df.Date_of_Journey.dt.month

            # Dropping "Date_of_Journey" column as it is not required anymore
            df.drop(columns=["Date_of_Journey"], inplace=True)

            # change data type of Dep_Time from object to datetime
            df.Dep_Time = pd.to_datetime(df.Dep_Time)

            # create two seperate columns named "Dep_hr" and "Dep_min"
            df["Dep_hr"] = df.Dep_Time.dt.hour
            df["Dep_min"] = df.Dep_Time.dt.minute

            # Dropping "Dep_Time" column as it is not required anymore
            df.drop(columns=["Dep_Time"], inplace=True)

            # change data type of Arrival_Time from object to datetime
            df.Arrival_Time = pd.to_datetime(df.Arrival_Time)

            # create two seperate columns named "Arrival_hr" and "Arrival_min"
            df["Arrival_hr"] = df.Arrival_Time.dt.hour
            df["Arrival_min"] = df.Arrival_Time.dt.minute

            # Dropping "Arrival_Time" column as it is not required anymore
            df.drop(columns=["Arrival_Time"], inplace=True)

            # we can convert all the values in Duration column into equivallent value in min
            def duration_in_min(dur):
                tt = 0
                for i in dur.split():
                    if 'h' in i:
                        tt += int(i[:-1])*60
                    if 'm' in i:
                        tt += int(i[:-1])
                return tt

            df.Duration = df.Duration.apply(duration_in_min)
            return df
        except Exception as e:
            FlightFareException(error_message=e, error_detail=sys)

    def encode_categorical_features(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            # Apply LabelEncoder on "Total_Stops" column
            df.Total_Stops = le.fit_transform(df["Total_Stops"])
            
            # Apply OneHotEncode on "Airline", "Source", "Destination" columns
            df = pd.get_dummies(df, columns=["Source", "Destination", "Airline"])   
            
            return df
        except Exception as e:
            print(e)
            FlightFareException(error_message=e, error_detail=sys)

    def encode_airline_df(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            # drop unnecessary features
            df.drop(columns=['Airline'], inplace=True)

            # OneHotEncoding on "Source", "Destination"
            df = pd.get_dummies(df, columns=["Source", "Destination"])

            # LabelEncoding on Total_Stops
            df.Total_Stops = le.fit_transform(df["Total_Stops"])
            return df
        except Exception as e:
            FlightFareException(e, sys)

    def initiate_transformation(self,)->artifact_entity.DataTransformationArtifact:
        try:
            logging.info("Loading Train and Test DF")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Columns before transformation: Train Data:{train_df.columns}")
            logging.info(f"Columns before transformation: Test Data:{test_df.columns}")

            useless = self.data_transformation_config.useless_features
            logging.info(f"Drop useless columns like {useless} if exists")
            for i in useless:
                if i in train_df.columns:
                    logging.info(f"{i} feature removed from train df")
                    train_df.drop(columns=[i], inplace=True)
                if i in test_df.columns:
                    logging.info(f"{i} feature removed from test df")
                    test_df.drop(columns=[i], inplace=True)

            # remove na values
            logging.info("Droping na values")
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            logging.info("Transforming all Numerical Features in Train and Test DF")
            train_df = self.transform_numerical_features(df=train_df)
            test_df = self.transform_numerical_features(df=test_df)

            # insert row if any category is missing in Categorcal features
            for col in ["Source", "Destination", "Airline"]:
                if len(train_df[col].unique()) > len(test_df[col].unique()):
                    for i in set(train_df[col].unique()).difference(list(test_df[col].unique())):
                        test_df.loc[len(test_df.index)] = [i, 'Kolkata', 'Banglore', 120, '0', 12, 8, 8, 30, 10, 30, 6000]
                        logging.info("1 row inserted")
            logging.info(f"new inserted rows: \n{test_df.iloc[-3:]}")

            airline_train_df = train_df.copy()
            airline_test_df = test_df.copy()
            
            # making data for individual model
            logging.info("Grouping Dataframe on the basis of Airline name")
            train_df_IndiGo = airline_train_df[airline_train_df.Airline == 'IndiGo']
            train_df_IndiGo = self.encode_airline_df(train_df_IndiGo)

            train_df_AirIndia = airline_train_df[airline_train_df.Airline == 'Air India']
            train_df_AirIndia = self.encode_airline_df(train_df_AirIndia)

            train_df_JetAirways = airline_train_df[airline_train_df.Airline == 'Jet Airways']
            train_df_JetAirways = self.encode_airline_df(train_df_JetAirways)

            train_df_SpiceJet = airline_train_df[airline_train_df.Airline == 'SpiceJet']
            train_df_SpiceJet = self.encode_airline_df(train_df_SpiceJet)

            train_df_Multiplecarriers = airline_train_df[airline_train_df.Airline == 'Multiple carriers']
            train_df_Multiplecarriers = self.encode_airline_df(train_df_Multiplecarriers)

            train_df_GoAir = airline_train_df[airline_train_df.Airline == 'GoAir']
            train_df_GoAir = self.encode_airline_df(train_df_GoAir)

            train_df_Vistara = airline_train_df[airline_train_df.Airline == 'Vistara']
            train_df_Vistara = self.encode_airline_df(train_df_Vistara)

            train_df_AirAsia = airline_train_df[airline_train_df.Airline == 'Air Asia']
            train_df_AirAsia = self.encode_airline_df(train_df_AirAsia)

            test_df_IndiGo = airline_test_df[airline_test_df.Airline == 'IndiGo']
            test_df_IndiGo = self.encode_airline_df(test_df_IndiGo)

            test_df_AirIndia = airline_test_df[airline_test_df.Airline == 'Air India']
            test_df_AirIndia = self.encode_airline_df(test_df_AirIndia)

            test_df_JetAirways = airline_test_df[airline_test_df.Airline == 'Jet Airways']
            test_df_JetAirways = self.encode_airline_df(test_df_JetAirways)

            test_df_SpiceJet = airline_test_df[airline_test_df.Airline == 'SpiceJet']
            test_df_SpiceJet = self.encode_airline_df(test_df_SpiceJet)

            test_df_Multiplecarriers = airline_test_df[airline_test_df.Airline == 'Multiple carriers']
            test_df_Multiplecarriers = self.encode_airline_df(test_df_Multiplecarriers)

            test_df_GoAir = airline_test_df[airline_test_df.Airline == 'GoAir']
            test_df_GoAir = self.encode_airline_df(test_df_GoAir)

            test_df_Vistara = airline_test_df[airline_test_df.Airline == 'Vistara']
            test_df_Vistara = self.encode_airline_df(test_df_Vistara)

            test_df_AirAsia = airline_test_df[airline_test_df.Airline == 'Air Asia']
            test_df_AirAsia = self.encode_airline_df(test_df_AirAsia)

            logging.info("Encoding all Categorical Features in Train and Test DF")
            train_df = self.encode_categorical_features(df=train_df)
            test_df = self.encode_categorical_features(df=test_df)

            logging.info(f"Columns after transformation: Train Data:\n{train_df.iloc[0:3]}")
            logging.info(f"Columns after transformation: Test Data:\n{test_df.iloc[0:3]}")

            logging.info(f"Create general_model folder in data_transformation folder if not available.")
            data_transformation_dir = os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            os.makedirs(data_transformation_dir, exist_ok=True)

            logging.info(f"Save Train and Test DF to transformed folder.")
            train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path, index=False, header=True)

            logging.info(f"Create individual airline folder in data_transformation folder if not available.")
            # Indigo
            IndiGo_dir = os.path.dirname(self.data_transformation_config.IndiGo_train_file_path)
            os.makedirs(IndiGo_dir, exist_ok=True)
            train_df_IndiGo.to_csv(self.data_transformation_config.IndiGo_train_file_path, index=False, header=True)
            test_df_IndiGo.to_csv(self.data_transformation_config.IndiGo_test_file_path, index=False, header=True)

            # AirIndia
            AirIndia_dir = os.path.dirname(self.data_transformation_config.AirIndia_train_file_path)
            os.makedirs(AirIndia_dir, exist_ok=True)
            train_df_AirIndia.to_csv(self.data_transformation_config.AirIndia_train_file_path, index=False, header=True)
            test_df_AirIndia.to_csv(self.data_transformation_config.AirIndia_test_file_path, index=False, header=True)

            # JetAirways
            JetAirways_dir = os.path.dirname(self.data_transformation_config.JetAirways_train_file_path)
            os.makedirs(JetAirways_dir, exist_ok=True)
            train_df_JetAirways.to_csv(self.data_transformation_config.JetAirways_train_file_path, index=False, header=True)
            test_df_JetAirways.to_csv(self.data_transformation_config.JetAirways_test_file_path, index=False, header=True)

            # SpiceJet
            SpiceJet_dir = os.path.dirname(self.data_transformation_config.SpiceJet_train_file_path)
            os.makedirs(SpiceJet_dir, exist_ok=True)
            train_df_SpiceJet.to_csv(self.data_transformation_config.SpiceJet_train_file_path, index=False, header=True)
            test_df_SpiceJet.to_csv(self.data_transformation_config.SpiceJet_test_file_path, index=False, header=True)

            # Multiplecarriers
            Multiplecarriers_dir = os.path.dirname(self.data_transformation_config.Multiplecarriers_train_file_path)
            os.makedirs(Multiplecarriers_dir, exist_ok=True)
            train_df_Multiplecarriers.to_csv(self.data_transformation_config.Multiplecarriers_train_file_path, index=False, header=True)
            test_df_Multiplecarriers.to_csv(self.data_transformation_config.Multiplecarriers_test_file_path, index=False, header=True)

            # GoAir
            GoAir_dir = os.path.dirname(self.data_transformation_config.GoAir_train_file_path)
            os.makedirs(GoAir_dir, exist_ok=True)
            train_df_GoAir.to_csv(self.data_transformation_config.GoAir_train_file_path, index=False, header=True)
            test_df_GoAir.to_csv(self.data_transformation_config.GoAir_test_file_path, index=False, header=True)

            # Vistara
            Vistara_dir = os.path.dirname(self.data_transformation_config.Vistara_train_file_path)
            os.makedirs(Vistara_dir, exist_ok=True)
            train_df_Vistara.to_csv(self.data_transformation_config.Vistara_train_file_path, index=False, header=True)
            test_df_Vistara.to_csv(self.data_transformation_config.Vistara_test_file_path, index=False, header=True)

            # AirAsia
            AirAsia_dir = os.path.dirname(self.data_transformation_config.AirAsia_train_file_path)
            os.makedirs(AirAsia_dir, exist_ok=True)
            train_df_AirAsia.to_csv(self.data_transformation_config.AirAsia_train_file_path, index=False, header=True)
            test_df_AirAsia.to_csv(self.data_transformation_config.AirAsia_test_file_path, index=False, header=True)



            logging.info(f"Prepearing artifact of DataTransformation")
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                                                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                                                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                                                IndiGo_train_file_path=self.data_transformation_config.IndiGo_train_file_path,
                                                AirIndia_train_file_path=self.data_transformation_config.AirIndia_train_file_path,
                                                JetAirways_train_file_path=self.data_transformation_config.JetAirways_train_file_path,
                                                SpiceJet_train_file_path=self.data_transformation_config.SpiceJet_train_file_path,
                                                Multiplecarriers_train_file_path=self.data_transformation_config.Multiplecarriers_train_file_path,
                                                GoAir_train_file_path=self.data_transformation_config.GoAir_train_file_path,
                                                Vistara_train_file_path=self.data_transformation_config.Vistara_train_file_path,
                                                AirAsia_train_file_path=self.data_transformation_config.AirAsia_train_file_path,
                                                IndiGo_test_file_path=self.data_transformation_config.IndiGo_test_file_path,
                                                AirIndia_test_file_path=self.data_transformation_config.AirIndia_test_file_path,
                                                JetAirways_test_file_path=self.data_transformation_config.JetAirways_test_file_path,
                                                SpiceJet_test_file_path=self.data_transformation_config.SpiceJet_test_file_path,
                                                Multiplecarriers_test_file_path=self.data_transformation_config.Multiplecarriers_test_file_path,
                                                GoAir_test_file_path=self.data_transformation_config.GoAir_test_file_path,
                                                Vistara_test_file_path=self.data_transformation_config.Vistara_test_file_path,
                                                AirAsia_test_file_path=self.data_transformation_config.AirAsia_test_file_path
                                                )

            logging.info(f"Data Transformation Artifact : {data_transformation_artifact}")
            return data_transformation_artifact  
        except Exception as e:
            raise FlightFareException(e, sys)
