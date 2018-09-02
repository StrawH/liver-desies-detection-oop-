import pandas as pd
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class LiverDesies:
    def __init__(self):
        self.__is_model_existed = False
        self.__data = None
        self.__model = None


        if os.path.isfile('Liverdetect.model'):
            self.__model = pickle.load(open("Liverdetect.model",'rb'))
            self.__is_model_existed = True
            print("the model is uploaded")

    def load_data_set (self, datset_name ):

        if not os.path.isfile(datset_name):
            raise ValueError ('please write the data set name with extension CSV')


        self.__data = pd.read_csv(datset_name)
        print("data set is loaded")



    def train (self,test_size=.2):

        if self.__data is None :
            raise ValueError ('make sure you loaded the data pls call load_data_set(data path)')

        self.__data = self.__data.dropna(how='any')
        self.__data['gender'] = self.__data['gender'].map({'Female':0,'Male':1})
        self.__data['is_patient'] = self.__data['is_patient'].map({1:0,2:1})
        self.OUTPUT = self.__data.pop('is_patient')
        self.INPUT = self.__data

        #split data to test it
        X_TRAIN, self.X_TEST, Y_TRANIN, self.Y_TEST = train_test_split(self.INPUT, self.OUTPUT, test_size=test_size)

        self.__model = SVC()
        self.__model.fit(X_TRAIN,Y_TRANIN)
        pickle.dump(self.__model,open('Liverdetect.model','wb'))
        self.__is_model_existed = True

        print("training is done and the model is saved" )



    def print_classification_result (self):
        Y_PRIDECTED = self.__model.predict(self.X_TEST)
        result = classification_report(self.Y_TEST, Y_PRIDECTED)
        print(result)


    def predict (self , feature):
        Data_frame = pd.DataFrame(self.__data)
        headers_number = list(Data_frame.keys()).__len__()
        fet_no = feature.__len__()  # features number of input data

        if self.__is_model_existed is True :
            headers_number = 10

        if not self.__is_model_existed  is True :
            raise ValueError ("tha model is not trained yet")

        if not fet_no == headers_number :
            raise ValueError ("features are not correct")

        if self.__model.predict([feature]) == 0:
            print("the predicted result : patient dosen't have a liver desies")

        else:
            print("the predicted result : patient have a liver desies")



