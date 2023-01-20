from flight_fare.entity import config_entity, artifact_entity
from flight_fare.exception import FlightFareException
from flight_fare.logger import logging
from flight_fare.config import TARGET_COLUMN
from flight_fare import utility
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import os, sys


class ModelTrainer:
    
    def __init__(self,
                    model_trainer_config=config_entity.ModelTrainingConfig,
                    data_transformation_artifact = artifact_entity.DataTransformationArtifact) -> None:
        try:
            logging.info(f"\n\n{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            FlightFareException(e, sys)

    def fine_tune(self,previous_model, x_train,y_train):
        try:   
            logging.info(f"Initiating GridSearchCV to get the best model")         
            grid = GridSearchCV(previous_model, self.model_trainer_config.param_distributions, n_jobs=-2)
            grid.fit(x_train,y_train)
            logging.info("HyperParameter Tuning is Completed.")
            return grid.best_estimator_

        except Exception as e:
            FlightFareException(e, sys)

    def train_model(self, train_df:pd.DataFrame, test_df:pd.DataFrame):
        try:
            # x-y split
            logging.info("Separate Independent and Dependent Features from Train and Test DF")
            x_train, y_train = train_df.drop(TARGET_COLUMN, axis=1), train_df[TARGET_COLUMN]
            x_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            print(x_train.shape, x_test.shape, sep='\n')
            # model training
            logging.info("training RandomForest Regressor Model")
            rf_reg =  RandomForestRegressor()
            rf_reg.fit(x_train,y_train)

            # Prediction
            y_pred = rf_reg.predict(x_test)

            # model performance
            train_score = rf_reg.score(x_train, y_train)
            test_score = rf_reg.score(x_test, y_test)
            logging.info(f"Train Score = {train_score}")
            logging.info(f"Test Score = {test_score}")
            logging.info(f"R2 Score: {metrics.r2_score(y_test, y_pred)}")

            # if test_score<self.model_trainer_config.expected_score:
            #     best_model = self.fine_tune(previous_model=rf_reg, x_train=x_train, y_train=y_train)
            #     # Prediction
            #     y_pred = best_model.predict(x_test)
            #     # best model performance
            #     train_score = best_model.score(x_train, y_train)
            #     test_score = best_model.score(x_test, y_test)
            #     logging.info(f"Train Score = {train_score}")
            #     logging.info(f"Test Score = {test_score}")
            #     logging.info(f"R2 Score: {metrics.r2_score(y_test, y_pred)}")
            #     return best_model, train_score, test_score

            # else:
            return rf_reg, train_score, test_score
        except Exception as e:
            raise FlightFareException(e, sys)

    def initiate_model_training(self,)->artifact_entity.ModelTrainingArtifact:
        try:
            # gen_pred_model
            logging.info("Model I : General Prediction model")
            logging.info("Loading Train and Test Data")
            gen_train_df = pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            gen_test_df = pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
            gen_model, gen_train_score, gen_test_score = self.train_model(train_df=gen_train_df, test_df=gen_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if gen_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {gen_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(gen_train_score-gen_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.gen_model_file_path, obj=gen_model)
            logging.info(f"{'-'*30}")

            # AirAsia_model
            logging.info("Model II : AirAsia model")
            logging.info("Loading Train and Test Data")
            AirAsia_train_df = pd.read_csv(self.data_transformation_artifact.AirAsia_train_file_path)
            AirAsia_test_df = pd.read_csv(self.data_transformation_artifact.AirAsia_test_file_path)
            AirAsia_model, AirAsia_train_score, AirAsia_test_score = self.train_model(train_df=AirAsia_train_df, test_df=AirAsia_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if AirAsia_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {AirAsia_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(AirAsia_train_score-AirAsia_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.AirAsia_model_file_path, obj=AirAsia_model)
            logging.info(f"{'-'*30}")


            # AirIndia_model
            logging.info("Model III : AirIndia model")
            logging.info("Loading Train and Test Data")
            AirIndia_train_df = pd.read_csv(self.data_transformation_artifact.AirIndia_train_file_path)
            AirIndia_test_df = pd.read_csv(self.data_transformation_artifact.AirIndia_test_file_path)
            AirIndia_model, AirIndia_train_score, AirIndia_test_score = self.train_model(train_df=AirIndia_train_df, test_df=AirIndia_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if AirIndia_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {AirIndia_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(AirIndia_train_score-AirIndia_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.AirIndia_model_file_path, obj=AirIndia_model)
            logging.info(f"{'-'*30}")


            # GoAir_model
            logging.info("Model IV : GoAir model")
            logging.info("Loading Train and Test Data")
            GoAir_train_df = pd.read_csv(self.data_transformation_artifact.GoAir_train_file_path)
            GoAir_test_df = pd.read_csv(self.data_transformation_artifact.GoAir_test_file_path)
            GoAir_model, GoAir_train_score, GoAir_test_score = self.train_model(train_df=GoAir_train_df, test_df=GoAir_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if GoAir_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {GoAir_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(GoAir_train_score-GoAir_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.GoAir_model_file_path, obj=GoAir_model)
            logging.info(f"{'-'*30}")


            # IndiGo_model
            logging.info("Model V : IndiGo model")
            logging.info("Loading Train and Test Data")
            IndiGo_train_df = pd.read_csv(self.data_transformation_artifact.IndiGo_train_file_path)
            IndiGo_test_df = pd.read_csv(self.data_transformation_artifact.IndiGo_test_file_path)
            IndiGo_model, IndiGo_train_score, IndiGo_test_score = self.train_model(train_df=IndiGo_train_df, test_df=IndiGo_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if IndiGo_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {IndiGo_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(IndiGo_train_score-IndiGo_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.IndiGo_model_file_path, obj=IndiGo_model)
            logging.info(f"{'-'*30}")


            # JetAirways_model
            logging.info("Model VI : JetAirways model")
            logging.info("Loading Train and Test Data")
            JetAirways_train_df = pd.read_csv(self.data_transformation_artifact.JetAirways_train_file_path)
            JetAirways_test_df = pd.read_csv(self.data_transformation_artifact.JetAirways_test_file_path)
            JetAirways_model, JetAirways_train_score, JetAirways_test_score = self.train_model(train_df=JetAirways_train_df, test_df=JetAirways_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if JetAirways_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {JetAirways_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(JetAirways_train_score-JetAirways_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.JetAirways_model_file_path, obj=JetAirways_model)
            logging.info(f"{'-'*30}")


            # Multiplecarriers_model
            logging.info("Model VII : Multiplecarriers model")
            logging.info("Loading Train and Test Data")
            Multiplecarriers_train_df = pd.read_csv(self.data_transformation_artifact.Multiplecarriers_train_file_path)
            Multiplecarriers_test_df = pd.read_csv(self.data_transformation_artifact.Multiplecarriers_test_file_path)
            Multiplecarriers_model, Multiplecarriers_train_score, Multiplecarriers_test_score = self.train_model(train_df=Multiplecarriers_train_df, test_df=Multiplecarriers_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if Multiplecarriers_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {Multiplecarriers_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(Multiplecarriers_train_score-Multiplecarriers_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.Multiplecarriers_model_file_path, obj=Multiplecarriers_model)
            logging.info(f"{'-'*30}")


            # SpiceJet_model
            logging.info("Model VIII : SpiceJet model")
            logging.info("Loading Train and Test Data")
            SpiceJet_train_df = pd.read_csv(self.data_transformation_artifact.SpiceJet_train_file_path)
            SpiceJet_test_df = pd.read_csv(self.data_transformation_artifact.SpiceJet_test_file_path)
            SpiceJet_model, SpiceJet_train_score, SpiceJet_test_score = self.train_model(train_df=SpiceJet_train_df, test_df=SpiceJet_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if SpiceJet_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {SpiceJet_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(SpiceJet_train_score-SpiceJet_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.SpiceJet_model_file_path, obj=SpiceJet_model)
            logging.info(f"{'-'*30}")


            # Vistara_model
            logging.info("Model IX : Vistara model")
            logging.info("Loading Train and Test Data")
            Vistara_train_df = pd.read_csv(self.data_transformation_artifact.Vistara_train_file_path)
            Vistara_test_df = pd.read_csv(self.data_transformation_artifact.Vistara_test_file_path)
            Vistara_model, Vistara_train_score, Vistara_test_score = self.train_model(train_df=Vistara_train_df, test_df=Vistara_test_df)

            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if Vistara_test_score<self.model_trainer_config.expected_score:
                # raise Exception
                logging.info(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {Vistara_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(Vistara_train_score-Vistara_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                # raise Exception
                logging.info(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utility.save_object(file_path=self.model_trainer_config.Vistara_model_file_path, obj=Vistara_model)
            logging.info(f"{'-'*30}")

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainingArtifact(gen_model_file_path=self.model_trainer_config.gen_model_file_path,
                                                                            IndiGo_model_file_path=self.model_trainer_config.IndiGo_model_file_path,
                                                                            AirAsia_model_file_path=self.model_trainer_config.AirAsia_model_file_path,
                                                                            Vistara_model_file_path=self.model_trainer_config.Vistara_model_file_path,
                                                                            GoAir_model_file_path=self.model_trainer_config.GoAir_model_file_path,
                                                                            Multiplecarriers_model_file_path=self.model_trainer_config.Multiplecarriers_model_file_path,
                                                                            SpiceJet_model_file_path=self.model_trainer_config.SpiceJet_model_file_path,
                                                                            JetAirways_model_file_path=self.model_trainer_config.JetAirways_model_file_path,
                                                                            AirIndia_model_file_path=self.model_trainer_config.AirIndia_model_file_path)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise FlightFareException(e, sys)