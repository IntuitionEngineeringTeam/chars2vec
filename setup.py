import sys
import subprocess
PY_VER = sys.version[0]
subprocess.call(["pip{:} install -r requirements.txt".format(PY_VER)], shell=True)

from setuptools import setup

setup(
    name='chars2vec',
    version='0.1.3',
    author='Vladimir Chikin',
    author_email='v4@intuition.engineering',
    packages=['chars2vec'],
    include_package_data=True,
    package_data={'chars2vec': ['trained_models/*']},
    description='Character-based language model based on RNN',
    maintainer='Intuition',
    maintainer_email='dev@intuition.engineering',
    url='',
    download_url='https://github.com/IntuitionEngineeringTeam/chars2vec/archive/master.zip',
    license='Apache License 2.0',
    long_description='The chars2vec language model is based on the symbolic representation of words \
                    – the model maps each word to a vector of a fixed length. Model tries map the \
                    most similar words to the most closed vectors. With current approach is possible \
                    to create an embedding in vector space for any sequence of characters. \
                    Chars2vec is based on TensorFlow deep neural network so it does not keep \
                    dictionary of embeddings and generates vector inplace using pretrained model.  \
                    There are pretrained models of dimensions 50, 100 and 150 for the English \
                    language. Library provides convenient user API to train model for arbitrary \
                    set of characters.',
    classifiers=['Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3']
)