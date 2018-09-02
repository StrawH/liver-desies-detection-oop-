#load the model class
from liver_class import LiverDesies

liver_model = LiverDesies()

#upload data set file 
liver_model.load_data_set('liver_data_set.csv')

#train your model with part of ypur data set , default = .2 or 20%
liver_model.train(.2)       

#print the accuracy of the model 
liver_model.print_classification_result()

#use the model with new data 
liver_model.predict([])




