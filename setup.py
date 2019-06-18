import sys
import subprocess
PY_VER = sys.version[0]
subprocess.call(["pip{:} install -r requirements.txt".format(PY_VER)], shell=True)

from setuptools import setup

setup(
    name='chars2vec',
    version='0.1.7',
    author='Vladimir Chikin',
    author_email='v4@intuition.engineering',
    packages=['chars2vec'],
    include_package_data=True,
    package_data={'chars2vec': ['trained_models/*']},
    description='Character-based word embeddings model based on RNN',
    maintainer='Intuition',
    maintainer_email='dev@intuition.engineering',
    url='https://github.com/IntuitionEngineeringTeam/chars2vec',
    download_url='https://github.com/IntuitionEngineeringTeam/chars2vec/archive/master.zip',
    license='Apache License 2.0',
    long_description='Chars2vec library could be very useful if you are dealing with the texts \
                        containing abbreviations, slang, typos, or some other specific textual dataset. \
                        Chars2vec language model is based on the symbolic representation of words – \
                        the model maps each word to a vector of a fixed length. \
                        These vector representations are obtained with a custom neural netowrk while \
                        the latter is being trained on pairs of similar and non-similar words. \
                        This custom neural net includes LSTM, reading sequences of characters in words, as its part. \
                        The model maps similarly written words to proximal vectors. \
                        This approach enables creation of an embedding in vector space for any sequence of characters.\
                        Chars2vec models does not keep any dictionary of embeddings, \
                        but generates embedding vectors inplace using pretrained model. \
                        There are pretrained models of dimensions 50, 100, 150, 200 and 300 for the English language.\
                        The library provides convenient user API to train a model for an arbitrary set of characters.',
    classifiers=['Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3']
)