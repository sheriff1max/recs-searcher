from setuptools import setup, find_packages
import os


def readme():
  with open('README.md', 'r', encoding='utf-8') as f:
    return f.read()

setup(
  name='recs-searcher',
  version='0.0.2',
  author='sheriff1max',
  author_email='kobelevmaxim48@gmail.com',
  description='Search engine and registry error corrector',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/sheriff1max/recs-searcher',
  packages=find_packages(),
  install_requires=[
    'fastapi==0.104.1',
    'chroma-hnswlib==0.7.3',
    'chromadb',
    'datasets==2.14.5',
    'faiss-cpu',
    'gensim',
    'sentence-transformers==2.2.2',
    'natasha==0.10.0',
    'numpy==1.26.0',
    'pandas==2.1.1',
    'pymorphy2==0.8',
    'pymorphy3==1.2.1',
    'scikit-learn==1.3.1',
    'scipy==1.11.3',
    'spacy==3.7.2',
    'thefuzz==0.20.0',
    'torch==2.1.0',
    'torchvision==0.16.0',
    'words2numsrus==0.1.0',
    'wordtodigits==1.0.2',
  ],
  classifiers=[
    'Programming Language :: Python :: 3',
  ],
  keywords='python searcher corrector faiss fasttext embeddings',
  project_urls={
    'Bug tracker': 'https://github.com/sheriff1max/recs-searcher/issues'
  },
  python_requires='>=3.11.5'
)
