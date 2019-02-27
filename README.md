# chars2vec

#### Character-based word embeddings model based on RNN


The chars2vec language model is based on the symbolic representation of words 
– the model maps each word to a vector of a fixed length. Model tries map the 
most similar words to the most closed vectors. With current approach is possible 
to create an embedding in vector space for any sequence of characters. 
Chars2vec is based on TensorFlow deep neural network so it does not keep 
dictionary of embeddings and generates vector inplace using pretrained model.  

There are pretrained models of dimensions 50, 100 and 150 for the English 
language. Library provides convenient user API to train model for arbitrary 
set of characters.  Read more details about the architecture of [Chars2vec: 
Character-based language model for handling real world texts with spelling 
errors and human slang](https://towardsdatascience.com).

#### Model available for Python 2.7 and 3.0+.

### Installation

<h5> 1. Build and install from source </h5>
Download project source and run in your command line

~~~shell
>> python setup.py install
~~~

<h5> 2. Via pip </h5>
Run in your command line

~~~shell
>> pip install chars2vec
~~~

### Usage

Function `chars2vec.load_model(str path)` initializes the model from directory 
and returns `chars2vec.Chars2Vec` object.
There are three pretrained English model with dimensions: 50, 100 and 150.
To load this pretrained models:

~~~python
import chars2vec

# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150'
c2v_model = chars2vec.load_model('eng_50')
~~~ 
Method `chars2vec.Chars2Vec.vectorize_words(words)` returns `numpy.ndarray` of shape `(n_words, dim)` with word embeddings.

~~~python
words = ['list', 'of', 'words']

# Create word embeddings
word_embeddings = c2v_model.vectorize_words(words)
~~~

### Training

Function `chars2vec.train_model(int emb_dim, training_set, model_chars)` 
creates and trains new chars2vec model and returns `chars2vec.Chars2Vec` object.

Parameter `emb_dim` is a dimension of the model. Parameter `model_chars`
is a list of chars for the model. Characters that are not in the `model_chars`
 list will be ignored by the model. 

Each element of the training dataset must be represented by a pair of words
and a target value that describes the proximity of words. 
Thus, each row of the list `training_set` should be like `*word_1* *word_2* *target_value*`.
Read more about the format of the training dataset and the method 
of generating in [article about chars2vec](https://towardsdatascience.com).

Function `chars2vec.save_model(c2v_model, str path_to_model)` saves the trained model to directory.


~~~python
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

# Create and train chars2vec model using given training data
my_c2v_model = chars2vec.train_model(dim, training_set, model_chars)

# Save your pretrained model
chars2vec.save_model(my_c2v_model, path_to_model)

# Load your pretrained model 
c2v_model = chars2vec.load_model(path_to_model)
~~~

Full code examples for usage and training models see in
`example_usage.py` and `example_training.py` files.


### Contact us

Website of our team [IntuitionEngineering](https://intuition.engineering).

Core developer email: v4@intuition.engineering.

Intuition dev email: dev@intuition.engineering.
