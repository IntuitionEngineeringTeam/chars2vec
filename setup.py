from setuptools import setup

setup(
    name='chars2vec',
    version='0.1.0',
    author='V. Chikin',
    author_email='v4@intuition.engineering',
    packages=['chars2vec'],
    package_data={'chars2vec': 'trained_models/*'},
    description='character-based language model based on RNN',
    url='',
    classifiers=['Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3']
)