from liver_class import LiverDisease

liver_obj = LiverDesies()

liver_obj.load_data_set('liver_data_set.csv')

liver_obj.train(.1)

liver_obj.print_classification_result()

liver_obj.predict([])





