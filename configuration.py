data_configuration = {
    'new_csv' : False, # create new csv file , True: create new , False: don't create new
    'csv_file' : "data/annotations_file.csv", # main csv file for all samples
    'csv_train' : "data/annot_train.csv", # splitted csv file for training data
    'csv_test' : "data/annot_test.csv", # splitted csv file for testing data
    'images_path' : "data/digits", 
    'splitting' : False, # True: split to training and testing, False: don't split anything!
    'split_ratio' : 0.2, # train-test ratio
    'images_subset' : 2000, # how many images from each label
    'show_sample' : False, 
    'show_df' : False,
    'data_info_sample' : False,
    'batch_size' : 32
}

model_configuration = {
    'lr' : 0.1,
    'epochs' : 1
}