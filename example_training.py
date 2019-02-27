import chars2vec


dim = 50

path_to_model = 'path/to/model/directory'

X_train = [('mecbanizing', 'mechanizing'), # similar words, target is equal 0
           ('dicovery', 'dis7overy'), # similar words, target is equal 0
           ('prot$oplasmatic', 'prtoplasmatic'), # similar words, target is equal 0
           ('copulateng', 'lzateful'), # not similar words, target is equal 1
           ('estry', 'evadin6'), # not similar words, target is equal 1
           ('cirrfosis', 'afear') # not similar words, target is equal 1
          ]

y_train = [0, 0, 0, 1, 1, 1]

model_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
               '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
               '=', '>', '?', '@', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
               'x', 'y', 'z']

# Create and train chars2vec model using given training data
my_c2v_model = chars2vec.train_model(dim, X_train, y_train, model_chars)

# Save pretrained model
chars2vec.save_model(my_c2v_model, path_to_model)

words = ['list', 'of', 'words']

# Load pretrained model, create word embeddings
c2v_model = chars2vec.load_model(path_to_model)
word_embeddings = c2v_model.vectorize_words(words)
