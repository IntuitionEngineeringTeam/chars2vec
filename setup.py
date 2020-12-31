from setuptools import setup


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setup(
    name="chars2vec",
    version="0.1.8",
    author="Vladimir Chikin",
    author_email="v4@intuition.engineering",
    packages=["chars2vec"],
    include_package_data=True,
    package_data={"chars2vec": ["trained_models/*"]},
    description="Character-based word embeddings model based on RNN",
    maintainer="Intuition",
    maintainer_email="dev@intuition.engineering",
    url="https://github.com/IntuitionEngineeringTeam/chars2vec",
    download_url="https://github.com/IntuitionEngineeringTeam/chars2vec/archive/master.zip",
    license="Apache License 2.0",
    long_description=readme,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
)