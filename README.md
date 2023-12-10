# recs-searcher (Registry error correction system - Searcher)

    pip install recs-searcher

Система способна решить следующие задачи:
1. Исправление реестровых ошибок пользовательского ввода при сравнении с базой данных;
2. Поиск схожих текстовых записей на пользовательский текст по базе данных.

Функциональные возможности:
1. обучение моделей для создания эмбеддингов (например, TfIDF, FastText, SentenceTransformer) на собственных данных;
2. быстрый поиск по базе данных (например, KNN, Faiss, Chroma-DB, TheFuzzSearch);
3. дообучение.

Реализованные модули:
1. api;
2. augmentation;
3. dataset;
4. models;
5. preprocessing;
6. similarity_search.

## Примеры применения
Пример для быстрого использования: [пример API](https://github.com/sheriff1max/recs-searcher/blob/master/api_example.ipynb)
