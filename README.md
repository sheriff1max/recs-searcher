<div align="center">

[![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/sheriff1max/recs-searcher/.github/workflows/python-app.yml)](https://github.com/sheriff1max/recs-searcher/actions/workflows/python-app.yaml)
[![Read the docs](https://img.shields.io/readthedocs/recs-searcher)](https://recs-searcher.readthedocs.io/ru/latest/)

[![PyPI](https://img.shields.io/pypi/v/recs-searcher?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/recs-searcher/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/recs-searcher?style=for-the-badge&color=blue)](https://pepy.tech/project/recs-searcher) 
<br>
</div>

# recs-searcher — библиотека для поиска похожих текстов
Библиотека позволяет находить похожие на пользовательский ввод тексты из датасета.

### Содержание
 1. [Проблематика](#problems)
 2. [Особенности библиотеки](#features)
 3. [Установка](#install)
 4. [Примеры применения](#examples)
 5. [Помощь в выборе/настройке алгоритмов](#help)

### Проблематика <a name="problems"></a>
Пользовательский ввод может содержать как орфографические, так и реестровые ошибки.

Рассмотрим самые частые ошибки:
- используются сокращения или полные формы слова: `«Литературный институт имени А.М. Горького»` || `«Литературный институт им. А.М. Горького»`;
- пропущены либо добавлены слова: `«Литературный институт имени А.М. Горького»` || `«Институт имени А.М.Горького»`;
- пропущены либо добавлены дополнительные символы: `«Сибирский федеральный университет»` || `«Сибрский федерааальный универ»`;
- слова могут быть в не правильном порядке: `Институт космических и информационных технологий` || `Институт информационных и космических технологий`.

Данные проблемы помогает решить разработанный модуль `recs-searcher (registry error correction system - searcher)`, основанный на известных NLP-алгоритмах.

### Особенности библиотеки: <a name="features"></a>
 - модуль универсален для любого датасета;
 - содержит API для использования библиотеки;
 - содержит множество подмодулей алгоритмов для оптимизации задачи, из которых строится pipeline (предобработка текста, модели для создания эмбеддингов, алгоритмы для эффективного сравнения эмбеддингов, аугментация текста для оценки обученного pipeline);
 - возможность интерпретировать результаты обученных pipeline;
 - масштабирование библиотеки благодаря имеющимся абстрактным классам.

### Установка <a name="install"></a>

```commandline
pip install recs-searcher
```

### Примеры применения <a name="examples"></a>

1. Соберём pipeline:
```python
from recs_searcher import (
    dataset,  # учебные датасеты
    preprocessing,  # предобработка текста
    embeddings,  # преобразование текста в эмбеддинги
    similarity_search,  # быстрые поисковики в пространстве эмбеддингов
    augmentation,  # аугментация текста для валидации пайплайнов
    explain,  # интерпретация сходства двух текстов
    api,  # Пайплайн
)

model_embedding = embeddings.CountVectorizerWrapperEmbedding(
    analyzer='char',
    ngram_range=(1, 2),
)

pipeline = api.Pipeline(
    dataset=['Красноярск', 'Москва', 'Владивосток'],
    preprocessing=[preprocessing.TextLower()],
    model=model_embedding,
    searcher=similarity_search.FaissSearch,
    verbose=True,
)
# Pipeline ready!
```

2. Найдём 3 схожих текстов в базе данных на пользовательский ввод "Красный ярск":
```python
pipeline.search('Красный ярск', 3, ascending=True)
# return: pandas.DataFrame
```

[Документация](https://recs-searcher.readthedocs.io/ru/latest/).

Более подробные примеры кода [API](https://github.com/sheriff1max/recs-searcher/blob/master/notebooks/tutorial_rus.ipynb).

Пример [WEB-интерфейса](https://github.com/sheriff1max/web-recs-searcher), в который внедрена данная библиотека.

### Помощь в выборе/настройке алгоритмов <a name="help"></a>

| Алгоритм сравнения базы данных с пользовательским вводом | Преимущества | Недостатки |
| :-----: | :---: | :---: |
| Мешок слов | Хорошо подходит для исправления реестровых ошибок при отсутствии орфографических ошибок. Также подходит для задачи поиска похожих текстов из датасета на ключевые слова. Даёт качество выше для больших текстов, чем для коротких. | При совершении всего одной орфографической ошибки в тексте или изменении формы слов перестаёт качественно выдавать ответы. Данный метод требователен к памяти при больших объёмах датасета. |
| Мешок символов | Хорошо подходит для разных задач: исправление реестровых и орфографических ошибок, а также поиск похожих текстов в датасете на ключевые слова. Не зависит от положения слов в тексте, а зависит только от их символов и N-грамм. Показывает чаще лучше качество и работает быстрее, чем другие алгоритмы в этом модуле. | Данный метод требователен к памяти при больших объёмах датасета. Требует тонкую настройку гиперпараметров для разных датасетов. Плохо работает на длинных текстах из-за того, что не учитывается контекст N-грамм символов. |
| Расстояние Левенштейна | Хорошо подходит для исправления орфографических ошибок в текстах, где мало слов. Метод малотребователен к памяти по сравнению с другими алгоритмами. | Крайне медленный метод. Медленно и менее качественно работает на длинных текстах, чем на коротких. |
| Трансформерные модели | Создаёт вектора текстов фиксированной длины, что при больших объёмах датасетов может сэкономить память. Подходит для поиска больших статей из набора данных по запросу. Учитывает не только символы, но и контекст и положение слов в текстах. Можно ускорить с помощью GPU. | Требует тонкую настройку гиперпараметров. |

### Автор
- [Кобелев Максим](https://github.com/sheriff1max) — автор и единственный разработчик.
