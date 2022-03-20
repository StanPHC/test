#crashes when one note with one word

#fastai==2.5.3
from fastai import *
from fastai.text import * 
from fastai.text.all import *

#utils
from utils import *

#general
import os

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



class ClassificationModel(object):

    def __init__(self):
        """
        A classifier for Medical/Clinical Patient Notes

        Attributes:
            classifier_type(int):
                - model_fall: Fall Event Detection Model
                - model_skin: Skin/Wound Incident Detection Model
        """
        self.storage = list()
        self.read_config()
    
    def read_config(self):
        """
        Read the config file, containing information w.r.t. the possible models
        """
        self.conf = pd.read_csv(os.path.join(os.getcwd(), "src", "config.csv"))
        return
        
    def load_data(self, data):
        """
        Split the data for each model and save it for prediction
        """
        self.dict_dfs = split_datasets(data, self.conf)
        #print(self.dict_dfs)
        
        return

    def classify(self):
        """
        Do all preprocessing steps on the data loaded previously through model.load_data(data),
        using the config file
        """
        for model, (df, iterator) in self.dict_dfs.items():
            if self.conf.loc[self.conf["modelname"] == model].sent_lvl.values[0] == 1:
                self.dict_dfs.get(model)[0] = split_sentences(df)
            if self.conf.loc[self.conf["modelname"] == model].preprocess.values[0] == 1:
                self.dict_dfs.get(model)[0] = preprocess(df)

        self.predict()
            
        return


    def predict(self):
        """
        Predict on all data
        """
        for model, (df, iterator) in self.dict_dfs.items():
            classifier = load_learner(os.path.join(os.getcwd(), "src", "model_" + model, "model.pkl"))
            dl = classifier.dls.test_dl(df["Text"])
            preds, target = classifier.get_preds(dl=dl)
            df["Preds"] = np.argmax(preds, axis =1)
            dct = convert_to_dict(df, model_name = model)
            self.add_to_storage(dct)


        return 

    def add_to_storage(self, data):
        """
        """
        for item in data:
            self.storage.append(item)

        return

    def get_predictions(self):
        """
        """
        print(len(self.storage))
        return({"result": self.storage})


