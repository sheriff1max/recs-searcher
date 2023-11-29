# recs-searcher (Registry error correction system - Searcher)
Система способна решить следующие задачи:
1. Исправление реестровых ошибок пользовательского ввода при сравнении с базой данных;
2. Поиск схожих текстовых записей на пользовательский текст по базе данных.

Функциональные возможности:
1. Обучение моделей для создания эмбеддингов (например, TfIDF, FastText, SentenceTransformer) на собственных данных;
2. Быстрый поиск по базе данных (KNN, Faiss, Chroma-DBб TheFuzzSearch);
3. Дообучение.

Реализованные модули:
1. api;
2. augmentation;
3. dataset;
4. models;
5. preprocessing;
6. similarity_search.

## Примеры применения
Пример для быстрого использования: https://github.com/sheriff1max/recs-searcher/blob/master/recs/notebooks/api.ipynb
