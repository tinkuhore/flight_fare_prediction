from flight_fare import utility
from flight_fare.entity import config_entity
from flight_fare.components import data_ingestion, data_validation, data_transformer, model_trainer, model_pusher

if __name__ == "__main__":
    # start training pipeline
    training_pipeline_config = config_entity.TrainingPipelineConfig()
    
    #data ingestion
    data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    print(data_ingestion_config.to_dict())
    data_ingestion = data_ingestion.DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    # data validation
    data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_validation = data_validation.DataValidation(data_validation_config=data_validation_config, data_ingesttion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()

    # data transformation
    data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
    data_transformation = data_transformer.DataTransformation(data_ingestion_artifact=data_ingestion_artifact, data_transformation_config=data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_transformation()

    # model training
    model_trainer_config = config_entity.ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
    model_training = model_trainer.ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
    model_trainer_artifact = model_training.initiate_model_training()

    # model pusher
    model_pusher_config = config_entity.ModelPusherConfig()
    model_push = model_pusher.ModelPusher(model_pusher_config=model_pusher_config,model_training_artifact=model_trainer_artifact)
    model_pusher_artifact = model_push.initiate_model_pusher()