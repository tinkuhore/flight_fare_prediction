from flight_fare.entity import config_entity, artifact_entity
from flight_fare import utility
from flight_fare.exception import FlightFareException
from flight_fare.logger import logging
from typing import Optional
from scipy.stats import ks_2samp
import os, sys
import pandas as pd

class DataValidation:
    
    def __init__(self,
                    data_validation_config=config_entity.DataValidationConfig,
                    data_ingesttion_artifact=artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"\n\n{'>'*20} Data Validation {'<'*20}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact = data_ingesttion_artifact
            self.validation_error = dict()
        except Exception as e:
            FlightFareException(error_message=e, error_detail=sys)


    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold
        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =====================================================================================
        returns Pandas DataFrame if atleast a single column is available after missing columns drop else None
        Will also append the test result in validation_report
        """
        try:
            
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            #selecting column name which contains null
            logging.info(f"Finding column name which contains null more than {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            if len(list(drop_column_names))>0:
                logging.info(f"Columns to drop: {list(drop_column_names)}")
                self.validation_error[report_key_name]=list(drop_column_names)
                df.drop(list(drop_column_names),axis=1,inplace=True)

            #return None no columns left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise FlightFareException(e, sys)

    def is_all_features_available(self,base_df:pd.DataFrame, 
                                    current_df: pd.DataFrame, report_key_name: str)->bool:
        
        """
        This function will check if all the required features for model training is present or not

        base_df: df on top of which the comparision will take place
        curren_df: train or test df from Data Ingestion Artifact
        =====================================================================================
        returns True if all required features are present else False
        Will also append the test result in validation_report if False
        """
        
        try:
            # initial validation
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_features=[]

            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_features.append(base_column)
                    logging.info(f"Column : {base_column} not available.")

            if len(missing_features)>0:
                self.validation_error[report_key_name]=missing_features
                logging.info("All required features are not available")
                return False
            return True
        except Exception as e:
            FlightFareException(error_message=e, error_detail=sys)

    def data_drift(self,base_df:pd.DataFrame, 
                                    current_df: pd.DataFrame, report_key_name: str) -> None:
        
        """
        This function will compare the underlying continuous distributions of continuous numerical features

        base_df: df on top of which the comparision will take place
        curren_df: train or test df from Data Ingestion Artifact
        =====================================================================================
        returns None
        Will also append the test result in validation_report
        """
        
        try:
            drift_report = dict()
            num_col_name = "Price"
            base_data,current_data = base_df[num_col_name],current_df[num_col_name]
            #Null hypothesis is that both column data drawn from same distrubtion
            
            logging.info(f"Hypothesis {num_col_name}: {base_data.dtype}, {current_data.dtype} ")
            same_distribution =ks_2samp(base_data,current_data)

            if same_distribution.pvalue>0.05:
                #We are accepting null hypothesis
                drift_report[num_col_name]={
                    "pvalues":float(same_distribution.pvalue),
                    "same_distribution": True
                }
            else:
                drift_report[num_col_name]={
                    "pvalues":float(same_distribution.pvalue),
                    "same_distribution":False
                }
            
            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            FlightFareException(error_message=e, error_detail=sys)

    def catagorical_value_mismatch(self,base_df:pd.DataFrame, 
                                    current_df: pd.DataFrame, report_key_name: str)->bool:
        
        """
        This function will check if all the required catagories are present in every Catagorical Features

        base_df: df on top of which the comparision will take place
        curren_df: train or test df from Data Ingestion Artifact
        =====================================================================================
        returns True if only required catagories are present else False
        Will also append the test result in validation_report
        """

        try:
            missing_cat = dict()
            cat_nom_columns = ["Airline", "Source", "Destination"]

            for col in cat_nom_columns:
                base_feature_unique = base_df[col].unique()
                current_feature_unique = current_df[col].unique()
                
                if len(base_feature_unique)>=len(current_feature_unique):
                    missing_values=[]
                    for f in base_feature_unique:
                        if f not in current_feature_unique:
                            missing_values.append(f)
                    missing_cat[col]=missing_values
                    logging.info(f"In {col} column missing catagories are {missing_values}")
                    
                elif len(base_feature_unique)<len(current_feature_unique):
                    missing_values=[]
                    for f in current_feature_unique:
                        if f not in base_feature_unique:
                            missing_values.append(f)
                    missing_cat[col]=missing_values
                    logging.info(f"In {col} column excess catagories are {missing_values}")
                self.validation_error[report_key_name]=missing_cat
        except Exception as e:
            FlightFareException(error_message=e, error_detail=sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info("Reading Base DF")
            base_df = pd.read_excel(self.data_validation_config.base_file_path)
            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="columns with missing values in Base df")

            logging.info("Loading Train and Test DF")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(df=train_df, report_key_name="Columns with missing values in Train DF")
            test_df = self.drop_missing_values_columns(df=test_df, report_key_name="Columns with missing values in Test DF")

            logging.info("Is all required features available in Train DF?")
            train_df_status = self.is_all_features_available(base_df=base_df, current_df=train_df, report_key_name="missing features within Train DF")
            
            logging.info("Is all required features available in Test DF?")
            test_df_status = self.is_all_features_available(base_df=base_df, current_df=test_df, report_key_name="missing features within Test DF")

            if train_df_status:
                logging.info("As all columns are available Train DF, now detecting Data drift in Target Column")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data drift within train df")
                logging.info("Now checking for value mismatch in categorical features in Train DF")
                self.catagorical_value_mismatch(base_df=base_df, current_df=train_df, report_key_name="categorical value mismatch in Train DF")


            if test_df_status:
                logging.info("As all columns are available Test DF, now detecting Data drift in Target column")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data drift within test df")
                logging.info("Now checking for value mismatch in categorical features in Test DF")
                self.catagorical_value_mismatch(base_df=base_df, current_df=test_df, report_key_name="categorical value mismatch in Test DF")
                

            logging.info("Write Validation Report in yaml file")
            utility.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)

            logging.info(f"Data Validation Artifact : {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            FlightFareException(e, sys)