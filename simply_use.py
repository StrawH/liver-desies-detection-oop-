from liver_class import LiverDesies

liver_obj = LiverDesies()

liver_obj.load_data_set('liver_data_set.csv')

liver_obj.train(.1)

liver_obj.print_classification_result()

liver_obj.predict([17,0,0.7,0.2,145,18,36,7.2,3.9,1.18])





