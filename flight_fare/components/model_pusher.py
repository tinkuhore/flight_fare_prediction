from flight_fare.entity import config_entity, artifact_entity
from flight_fare.logger import logging
from flight_fare.exception import FlightFareException
from flight_fare.utility import load_object, save_object
import os, sys

class ModelPusher:

    def __init__(self, 
                model_pusher_config:config_entity.ModelPusherConfig, 
                model_training_artifact:artifact_entity.ModelTrainingArtifact):
        try:
            logging.info(f"\n\n{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.model_training_artifact = model_training_artifact
        except Exception as e:
            FlightFareException(e, sys)


    def initiate_model_pusher(self):
        try:
            # load models
            logging.info("Loading models from Model Training Artifact")
            gen_model = load_object(self.model_training_artifact.gen_model_file_path)
            IndiGo_model = load_object(self.model_training_artifact.IndiGo_model_file_path)
            AirAsia_model = load_object(self.model_training_artifact.AirAsia_model_file_path)
            Vistara_model = load_object(self.model_training_artifact.Vistara_model_file_path)
            GoAir_model = load_object(self.model_training_artifact.GoAir_model_file_path)
            Multiplecarriers_model = load_object(self.model_training_artifact.Multiplecarriers_model_file_path)
            SpiceJet_model = load_object(self.model_training_artifact.SpiceJet_model_file_path)
            JetAirways_model = load_object(self.model_training_artifact.JetAirways_model_file_path)
            AirIndia_model = load_object(self.model_training_artifact.AirIndia_model_file_path)

            # check if any dir already exists or not
            list_dir = os.listdir(self.model_pusher_config.saved_model_dir)
            if len(list_dir)==0:
                saved_models_path = os.path.join(self.model_pusher_config.saved_model_dir,f"{0}")
            else:
                latest_dir_num = int(max(list_dir))+1
                saved_models_path = os.path.join(self.model_pusher_config.saved_model_dir,f"{latest_dir_num}")

            # save objects
            logging.info("Saving models in saved_model dir")
            save_object(file_path=os.path.join(saved_models_path,"gen_pred_model.pkl"), obj=gen_model)
            save_object(file_path=os.path.join(saved_models_path,"IndiGo.pkl"), obj=IndiGo_model)
            save_object(file_path=os.path.join(saved_models_path,"AirAsia_mode.pkl"), obj=AirAsia_model)
            save_object(file_path=os.path.join(saved_models_path,"Vistara_model.pkl"), obj=Vistara_model)
            save_object(file_path=os.path.join(saved_models_path,"GoAir_model.pkl"), obj=GoAir_model)
            save_object(file_path=os.path.join(saved_models_path,"Multiplecarriers_model.pkl"), obj=Multiplecarriers_model)
            save_object(file_path=os.path.join(saved_models_path,"SpiceJet_model.pkl"), obj=SpiceJet_model)
            save_object(file_path=os.path.join(saved_models_path,"JetAirways_model.pkl"), obj=JetAirways_model)
            save_object(file_path=os.path.join(saved_models_path,"AirIndia_model.pkl"), obj=AirIndia_model)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(saved_model_dir=saved_models_path)
            logging.info(f"Model Pusher Artifact : {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            FlightFareException(e, sys)