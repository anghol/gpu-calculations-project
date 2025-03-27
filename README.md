# Проект по курсу "Вычисления на GPU"

## Задача: реализовать параллельную версию метода Симпсона для вычисления значения интеграла.

## Структура файлов проекта:
- `poisson_integral.cu` - первая (пробная) программа для экспериментов с вычислением интеграла Пуассона.
- `simpson_method.cu` - программа для CUDA для экспериментов, производится замер времени работы метода на CPU и GPU и вычисление коэффициента ускорения.
- `client.py` - клиент, который считывает json-файл с примерами и запускает выполнение расчетов на разных GPU посредством POST-запросов к серверам, запущенным на разных машинах.
- `server.py` - Flask-приложение, которое запускается на машине и ожидает от клиента POST-запросы для вычислений. Производит параллельные вычисления на GPU с помощью CUDA.
- `config.json` - файл с настройками: список серверов, список примеров для вычисления (текстовая запись подынтегральной функции как она была бы записана кодом на С, пределы интегрирования, размер блока).
- `run.sh` - bash-скрипт для компиляции и запуска программы `simpson_method.cu`.

## Запуск:
1. Запустить `server.py` на машинах с видеокартой.
2. Прописать url-адреса машин в файле `config.json`.
3. Прописать примеры задач в файле `config.json` с указанием текстового выражения функции (как бы она была корректно записана на языке С), пределов интегрирования и размера блока.
4. Запустить `client.py` для вычислений и посмотреть на результат выполнения в терминале.

## Детали `server.py`:
В коде этого файла есть переменная `cuda_code`, в которую в виде строки записан код для CUDA. В процессе работы сервер получает от клиента POST-запрос, в котором передается пример для вычисления в виде json-параметра. В функции `calculate` происходит чтение полученного json и извлечение данных: текстовое выражение функции, пределы интегрирования и размер блока. Текстовое выражение подынтегральной функции затем конкатенируется в виде \_\_host\_\_ \_\_device\_\_ функции к строковой переменной `cuda_code`. Таким образом, создается полный код для данной задачи для CUDA.