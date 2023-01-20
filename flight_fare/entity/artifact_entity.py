from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    frature_store_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    IndiGo_train_file_path:str
    AirIndia_train_file_path:str
    JetAirways_train_file_path:str
    SpiceJet_train_file_path:str
    Multiplecarriers_train_file_path:str
    GoAir_train_file_path:str
    Vistara_train_file_path:str
    AirAsia_train_file_path:str
    IndiGo_test_file_path:str
    AirIndia_test_file_path:str
    JetAirways_test_file_path:str
    SpiceJet_test_file_path:str
    Multiplecarriers_test_file_path:str
    GoAir_test_file_path:str
    Vistara_test_file_path:str
    AirAsia_test_file_path:str
    

@dataclass
class ModelTrainingArtifact:
    gen_model_file_path:str
    IndiGo_model_file_path:str
    AirAsia_model_file_path:str
    Vistara_model_file_path:str
    GoAir_model_file_path:str
    Multiplecarriers_model_file_path:str
    SpiceJet_model_file_path:str
    JetAirways_model_file_path:str
    AirIndia_model_file_path:str

class ModelEvaluationArtifact:...

@dataclass
class ModelPusherArtifact:
    saved_model_dir:str