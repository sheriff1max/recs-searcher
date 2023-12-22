from setuptools import setup, find_packages
import os


def readme():
  with open('README.md', 'r', encoding='utf-8') as f:
    return f.read()

setup(
  name='recs-searcher',
  version='0.0.4',
  author='sheriff1max',
  author_email='kobelevmaxim48@gmail.com',
  description='Search engine and registry error corrector',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/sheriff1max/recs-searcher',
  packages=find_packages(),
  package_data={
    'recs_searcher': ['dataset/data/*.csv'],
  },
  install_requires=[
    'fastapi==0.104.1',
    'chroma-hnswlib',
    'chromadb',
    'datasets',
    'faiss-cpu',
    'gensim',
    'sentence-transformers',
    'pymorphy3',
    'scikit-learn',
    'spacy',
    'thefuzz',
    'torch',
    'torchvision',
  ],
  classifiers=[
    'Programming Language :: Python :: 3',
  ],
  keywords='python searcher corrector faiss fasttext embeddings',
  project_urls={
    'Bug tracker': 'https://github.com/sheriff1max/recs-searcher/issues'
  },
  python_requires='>=3.11'
)
