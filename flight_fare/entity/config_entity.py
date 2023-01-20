import os, sys
from datetime import datetime
from flight_fare.exception import FlightFareException

FILE_NAME = "flight_fare.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifact", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            FlightFareException(e, sys)
        

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = "ffp"
            self.collection_name = "flight_fare"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            FlightFareException(e, sys)

    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            FlightFareException(e, sys)

class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
            self.base_file_path = os.path.join("archive/Data_Train.xlsx")
            self.missing_threshold: float = 0.2
        except Exception as e:
            FlightFareException(e, sys)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.useless_features = ["Route", "Additional_Info"]
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
            self.transformed_train_file_path = os.path.join(self.data_transformation_dir, "general_model", TRAIN_FILE_NAME)
            self.transformed_test_file_path = os.path.join(self.data_transformation_dir, "general_model", TEST_FILE_NAME)
            # self.transformed_object_path = os.path.join(self.data_transformation_dir, "transformed", "transfromer.pkl")

            self.IndiGo_train_file_path = os.path.join(self.data_transformation_dir, "IndiGo_model", TRAIN_FILE_NAME)
            self.IndiGo_test_file_path = os.path.join(self.data_transformation_dir, "IndiGo_model", TEST_FILE_NAME)

            self.AirIndia_train_file_path = os.path.join(self.data_transformation_dir, "AirIndia_model", TRAIN_FILE_NAME)
            self.AirIndia_test_file_path = os.path.join(self.data_transformation_dir, "AirIndia_model", TEST_FILE_NAME)

            self.JetAirways_train_file_path = os.path.join(self.data_transformation_dir, "JetAirways_model", TRAIN_FILE_NAME)
            self.JetAirways_test_file_path = os.path.join(self.data_transformation_dir, "JetAirways_model", TEST_FILE_NAME)

            self.SpiceJet_train_file_path = os.path.join(self.data_transformation_dir, "SpiceJet_model", TRAIN_FILE_NAME)
            self.SpiceJet_test_file_path = os.path.join(self.data_transformation_dir, "SpiceJet_model", TEST_FILE_NAME)

            self.Multiplecarriers_train_file_path = os.path.join(self.data_transformation_dir, "Multiplecarriers_model", TRAIN_FILE_NAME)
            self.Multiplecarriers_test_file_path = os.path.join(self.data_transformation_dir, "Multiplecarriers_model", TEST_FILE_NAME)

            self.GoAir_train_file_path = os.path.join(self.data_transformation_dir, "GoAir_model", TRAIN_FILE_NAME)
            self.GoAir_test_file_path = os.path.join(self.data_transformation_dir, "GoAir_model", TEST_FILE_NAME)

            self.Vistara_train_file_path = os.path.join(self.data_transformation_dir, "Vistara_model", TRAIN_FILE_NAME)
            self.Vistara_test_file_path = os.path.join(self.data_transformation_dir, "Vistara_model", TEST_FILE_NAME)

            self.AirAsia_train_file_path = os.path.join(self.data_transformation_dir, "AirAsia_model", TRAIN_FILE_NAME)
            self.AirAsia_test_file_path = os.path.join(self.data_transformation_dir, "AirAsia_model", TEST_FILE_NAME)
        except Exception as e:
            FlightFareException(e, sys)

class ModelTrainingConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
            self.gen_model_file_path = os.path.join(self.model_trainer_dir, "models", "gen_pred_model.pkl")
            self.IndiGo_model_file_path = os.path.join(self.model_trainer_dir, "models", "IndiGo.pkl")
            self.AirAsia_model_file_path = os.path.join(self.model_trainer_dir, "models", "AirAsia.pkl")
            self.Vistara_model_file_path = os.path.join(self.model_trainer_dir, "models", "Vistara.pkl")
            self.GoAir_model_file_path = os.path.join(self.model_trainer_dir, "models", "GoAir.pkl")
            self.Multiplecarriers_model_file_path = os.path.join(self.model_trainer_dir, "models", "Multiplecarriers.pkl")
            self.SpiceJet_model_file_path = os.path.join(self.model_trainer_dir, "models", "SpiceJet.pkl")
            self.JetAirways_model_file_path = os.path.join(self.model_trainer_dir, "models", "JetAirways.pkl")
            self.AirIndia_model_file_path = os.path.join(self.model_trainer_dir, "models", "AirIndia.pkl")
            self.expected_score = 0.6
            self.overfitting_threshold = 0.4
            self.param_distributions = {'max_depth': list(range(5,55,5)),
                        'max_features': ['log2', 'sqrt'],
                        'min_samples_leaf': list(range(1,6)),
                        'min_samples_split': list(range(1,100,2)),
                        'n_estimators': list(range(100,1300,100))}
        except Exception as e:
            FlightFareException(e, sys)

class ModelEvaluationConfig:...

class ModelPusherConfig:
    def __init__(self,):
        try:
            self.saved_model_dir = os.path.join("saved_models")
        except Exception as e:
            FlightFareException(e, sys)