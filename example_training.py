import chars2vec


dim = 50

path_to_model = 'path/to/model/directory'
path_to_training_set = 'path/to/txt/file/with/training/set'
training_set = open(path_to_training_set, 'r').readlines()

model_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
               '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
               '=', '>', '?', '@', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
               'x', 'y', 'z']

my_c2v_model = chars2vec.train_model(dim, training_set, model_chars)
chars2vec.save_model(my_c2v_model, path_to_model)

words = ['list', 'of', 'words']

c2v_model = chars2vec.load_model(path_to_model)
word_embeddings = c2v_model.vectorize_words(words)
