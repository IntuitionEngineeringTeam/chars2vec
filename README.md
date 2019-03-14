# chars2vec

#### Character-based word embeddings model based on RNN


Chars2vec library could be very useful if you are dealing with the texts 
containing abbreviations, slang, typos, or some other specific textual dataset. 
Chars2vec language model is based on the symbolic representation of words – 
the model maps each word to a vector of a fixed length. 
These vector representations are obtained with a custom neural netowrk while 
the latter is being trained on pairs of similar and non-similar words. 
This custom neural net includes LSTM, reading sequences of characters in words, as its part. 
The model maps similarly written words to proximal vectors. 
This approach enables creation of an embedding in vector space for any sequence of characters. 
Chars2vec models does not keep any dictionary of embeddings, 
but generates embedding vectors inplace using pretrained model. 

There are pretrained models of dimensions 50, 100, 150, 200 and 300 for the English language.
The library provides convenient user API to train a model for an arbitrary set of characters. 
Read more details about the architecture of [Chars2vec: 
Character-based language model for handling real world texts with spelling 
errors and human slang](https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d) in Hacker Noon.

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
There are 5 pretrained English model with dimensions: 50, 100, 150, 200 and 300.
To load this pretrained models:

~~~python
import chars2vec

# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150', 'eng_200', 'eng_300'
c2v_model = chars2vec.load_model('eng_50')
~~~ 
Method `chars2vec.Chars2Vec.vectorize_words(words)` returns `numpy.ndarray` of shape `(n_words, dim)` with word embeddings.

~~~python
words = ['list', 'of', 'words']

# Create word embeddings
word_embeddings = c2v_model.vectorize_words(words)
~~~

### Training

Function `chars2vec.train_model(int emb_dim, X_train, y_train, model_chars)` 
creates and trains new chars2vec model and returns `chars2vec.Chars2Vec` object.

Parameter `emb_dim` is a dimension of the model. 

Parameter `X_train` is a list or numpy.ndarray of word pairs.
Parameter `y_train` is a list or numpy.ndarray of target values that describe the proximity of words.

Training set (`X_train`, `y_train`) consists of pairs of "similar" and "not similar" words; 
a pair of "similar" words is labeled with 0 target value, and a pair of "not similar" with 1. 

Parameter `model_chars` is a list of chars for the model.
Characters which are not in the `model_chars`
list will be ignored by the model. 

Read more about chars2vec training and generation of training dataset in 
[article about chars2vec](https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d).

Function `chars2vec.save_model(c2v_model, str path_to_model)` saves the trained model 
to the directory.


~~~python
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
