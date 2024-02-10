import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # it will take trained model pkl file
            model_path = os.path.join("artifacts", "model.pkl")
            # it will be responsible for categorical and scaling data
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            logging.info("Before Loading model and preprocessor file")
            print("Before Loading")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            logging.info("After Loading model and preprocessor file")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


"""
This CustomData class will be responsible for mapping all the inputs that we are giving from html page to the backend to this particular value
"""


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        # Creating a variable
        self.gender = gender,
        self.race_ethnicity = race_ethnicity,
        self.parental_level_of_education = parental_level_of_education,
        self.lunch = lunch,
        self.test_preparation_course = test_preparation_course,
        self.reading_score = reading_score,
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score,
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
