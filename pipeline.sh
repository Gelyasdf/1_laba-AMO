#!/bin/bash

#запускаем скрипт для создания данных
python data_creation.py

#скрипт для предобработки данных
python model_preprocessing.py

#скрипт для подготовки модели
python model_preparation.py

#скрипт для тестирования модели
python model_testing.py