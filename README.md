# Magnum — классификация овощей по фотографии (Track A, ResNet18)

Решение тестового задания **Junior Computer Vision / Machine Learning**: классификатор шести видов овощей на базе **ResNet18 (PyTorch)**.

**Основной интерфейс для проверки:** программа `predict.py` получает папку изображений и сохраняет CSV с тремя колонками — `filename,label,confidence`. Для запуска используется **готовая модель Stage B** из `model/`. **Повторное обучение, Kaggle, Google Colab, Docker и GPU не нужны.** Инференс выполняется на CPU без сетевых запросов после установки зависимостей.

Поддерживаемые классы: `Capsicum`, `Carrot`, `Cucumber`, `Potato`, `Raddish`, `Squash`. Написание `Raddish` сохранено из исходного датасета.

## 1. Быстрый запуск (Quick Start)

### Шаг 1. Получить файлы и установить зависимости

Склонируйте репозиторий через Git или скачайте его через **Code → Download ZIP**, распакуйте архив и откройте терминал **в корне проекта**, где находятся `predict.py` и `requirements.txt`.

Требуется **Python 3.10 или новее**. Проверьте доступную версию: `python3 --version` (на Windows: `py --version`). Если `python3` на macOS указывает на системный Python 3.9, выберите отдельно установленную версию 3.12 или новее.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Если установленный интерпретатор называется `python3.12`, используйте `python3.12 -m venv .venv` вместо первой команды.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Команды Windows приведены как вариант установки, отдельно в этом проекте не тестировались. Используйте установленную совместимую версию Python, если она отличается от 3.12.

### Шаг 2. Проверить запуск на готовом примере

В корне проекта выполните:

```bash
python predict.py --input examples --output preds.csv
```

Программа обработает `examples/bench_photo.jpg` и создаст `preds.csv` в текущем каталоге. Внутри будет заголовок и строка предсказания, например:

```csv
filename,label,confidence
bench_photo.jpg,Squash,0.999954
```

Это **пример ранее полученного вывода**, а не гарантированное побитовое совпадение при других версиях библиотек и не показатель точности на независимом тесте.

### Шаг 3. Запустить на своих изображениях

Поместите изображения в папку `photos/` (или укажите путь к любой другой папке):

```bash
python predict.py --input photos --output predictions.csv
```

В `predictions.csv` будет по одной строке на каждое изображение и ровно три колонки:

| Колонка | Значение |
|---|---|
| `filename` | Имя файла или **относительный путь** внутри входной папки |
| `label` | Один из шести поддерживаемых классов |
| `confidence` | Максимальная оценка softmax, число от 0 до 1 |

Поддерживаются `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`; вложенные папки обрабатываются рекурсивно. Если во входной папке нет поддерживаемых изображений или файл не удаётся прочитать, программа сообщает об ошибке. Один и тот же файл весов загружается один раз за запуск.

`confidence` — **некалиброванная softmax-оценка, не гарантированная вероятность правильности**. Режим `unknown` не реализован: даже для изображения без этих овощей будет выбран один из шести классов.

### Дополнительные команды (необязательно)

```bash
# Вывести предсказание для одной фотографии в терминал
python predict.py --image examples/bench_photo.jpg --device cpu

# Проверить CLI и формат CSV на двух автоматически созданных изображениях
python -m unittest discover -s tests -v

# Явно выбрать исследовательский checkpoint Stage A вместо Stage B
python predict.py --input examples --output preds_stage_a.csv --weights model/resnet18_quality_best.pth
```

Интерфейс `--input ... --output ...` — основной для проверки. Параметр `--device` по умолчанию равен `cpu`; CUDA опциональна, если доступна. **При инференсе скачивание ImageNet-весов и обращение к сетевым API отсутствуют.** Первоначальная установка Python-пакетов может потребовать интернет.

### Окружения, в которых выполнялись проверки

- **macOS:** локальные запуск и CLI-тест с Python **3.12.10**, PyTorch **2.15.0**, torchvision **0.29.2**, Pillow **12.3.0**.
- **Linux x86-64:** проверки CPU-инференса и benchmark с Python **3.13.5**, PyTorch **2.10.0+cpu**, torchvision **0.25.0+cpu**.

Диапазоны зависимостей указаны в `requirements.txt`; отдельная проверка всех возможных комбинаций Python и пакетов не проводилась. Значения скорости ниже относятся только к указанной Linux-среде.

## 2. Модели и результаты оценки

В проекте **две внутренние стадии** (не путать с Track A работодателя):

| Стадия | Что делалось | Данные | Как интерпретировать результаты |
|---|---|---|---|
| **Stage A** | Сравнение **Quality** и **CNN Diversity** с общими Validation/Test | Train: 2 488 / 1 798; Validation: 552; Test: 683 | Quality выбрана по Validation macro-F1 **0,9574** против **0,9182**; затем однократно оценена на Test: **667/683**, accuracy **97,66%**, macro-F1 **0,9159** |
| **Stage B** | Новая ResNet18 от предобученных ImageNet-весов, обученная на всём качественном корпусе | **3 723** изображения, включая бывшие Validation/Test | Это **модель по умолчанию для `predict.py`**. Независимой Test-оценки у неё **нет** |

Чекпоинты:

- `model/resnet18_stage_b_full_quality_epoch5.pth` — **основная модель для инференса**.
- `model/resnet18_quality_best.pth` — сохранённая лучшая **Stage A Quality**, для которой получены представленные Validation/Test-метрики.

**Нельзя приписывать Stage B точность 97,66%:** прежние тестовые фотографии вошли в обучение Stage B. Значение 100% Train accuracy на последней эпохе Stage B также не является независимой оценкой.

Полные материалы: [двухстраничный отчёт](REPORT.pdf), [текст отчёта](REPORT.md), [исследование данных](analysis/DATA_ANALYSIS.md), [матрица ошибок Test](results/stage_a/test_confusion_matrix.png), [Grad-CAM на восьми примерах Stage A](results/stage_a/test_gradcam_examples.png). Индивидуальные предсказания и classwise-метрики находятся в `results/stage_a/`.

## 3. Скорость CPU и интерпретация

Воспроизвести замер на текущем компьютере:

```bash
python benchmark_cpu.py --image examples/bench_photo.jpg --runs 20 --warmup 3 --threads 4
```

На контрольной **Linux x86-64 CPU-среде** (JPEG 212×211, 4 потока, 3 прогрева и 20 измерений) получены: **медиана 34,09 мс/изображение**, среднее 47,05 мс, P90 71,65 мс. Измерение включает чтение изображения, preprocessing и forward pass **после загрузки модели**. Первоначальная загрузка весов занимала отдельно около 0,241 с. Файл исходных результатов: `results/stage_b/cpu_latency_benchmark.json`.

`examples/bench_photo.jpg` — пример, происходящий из визуализации исходного Kaggle-датасета. Это **smoke test и benchmark**, а не независимое измерение accuracy и не фотография из магазина. Скорость на оборудовании Magnum может отличаться.

Дополнительная диагностическая визуализация для отдельной фотографии:

```bash
python -m pip install numpy matplotlib
python gradcam.py --image examples/bench_photo.jpg --output gradcam.png --device cpu
```

Grad-CAM показывает области, связанные с оценкой предсказанного класса; это **диагностическая визуализация**, а не причинное доказательство решения сети.

## 4. Структура репозитория

| Путь | Назначение |
|---|---|
| `predict.py` | Основной CLI для распознавания папки изображений на CPU |
| `model/` | Два локальных PyTorch-чекпоинта Stage A и Stage B |
| `requirements.txt` | Минимальные зависимости инференса |
| `tests/test_predict_cli.py` | Автоматическая проверка запуска и структуры выходного CSV (не accuracy-тест) |
| `examples/` | Контрольная фотография для быстрого запуска и benchmark |
| `benchmark_cpu.py` | Измерение задержки одного CPU-предсказания |
| `REPORT.pdf`, `REPORT.md` | Краткий отчёт и его Markdown-версия |
| `results/` | Метрики, предсказания, графики, confusion matrices, Grad-CAM и результаты CPU-проверок |
| `split_manifests/` | Зафиксированные составы Train, Validation и Test для Stage A |
| `analysis/` | Исследование датасета, таблицы, исходный исследовательский ноутбук и скрипты сверки |
| `training/` | Код, настройки и история обучения Stage A / Stage B |
| `evaluate.py` | Повторная оценка **Stage A** на исходных JPG и split V3 |
| `gradcam.py` | Построение Grad-CAM для одного изображения |

## 5. Воспроизведение экспериментов (необязательно)

**Этот раздел не нужен для запуска готового классификатора.** Он предназначен для технического аудита, перепроверки таблиц и возможного повторного обучения. Исходные JPG не включены в репозиторий (около 1,7 ГБ).

**Источник данных:** [Kaggle — preethi866/vegetable-dataset, версия 1](https://www.kaggle.com/datasets/preethi866/vegetable-dataset). Использован каталог `Dataset/Dataset` с шестью папками классов, а не готовый авторский `final_dataset`: в последнем обнаружены пересечения одинаковых изображений между выборками. По исследовательским материалам лицензия исходного датасета указана как Apache 2.0.

### 5.1. Сверить сохранённые данные без JPG и GPU

Из корня проекта:

```bash
python analysis/verify_split.py
python analysis/verify_research.py
```

Скрипты проверяют размеры V3, состав Train-only CNN Diversity, пересечения префиксных групп, список 156 исключённых по качеству фото и сохранённые Validation/Test-предсказания. Проверки используют замороженные CSV/NPY, **а не перечитывают исходные пиксели 3 879 изображений**. В `analysis/research_tables/` находятся 21 уникальный исследовательский CSV/NPY и `MANIFEST.json` с контрольными суммами.

**Важное уточнение о втором Train:** историческое имя `split_manifests/stage_a_train_cnn_balanced_v3.csv` не отражает метод отбора. Файл содержит **1 798** фотографий, выбранных **только внутри Train V3** методом CNN Diversity: квоты по `(class, shooting)`, typical-first и greedy farthest-first по косинусному сходству 512-мерных эмбеддингов. Все **52** Train-фото без заранее вычисленных эмбеддингов сохранены. Старая глобальная CNN Balanced-политика с **1 033** предложенными исключениями — **другой, более ранний эксперимент**, а не состав второго Train. Различия описаны в `analysis/RECONCILED_PROJECT_HISTORY.md`.

### 5.2. Повторно оценить Stage A (исходные JPG обязательны)

```bash
python -m pip install -r training/requirements-training.txt
python evaluate.py --data-root /путь/к/Dataset/Dataset --split test --device cpu
```

По умолчанию используется `model/resnet18_quality_best.pth`. Скрипт не позволяет выдать оценку Stage B на прежнем Test за независимую. Небольшие числовые отличия возможны из-за окружения и устройства. Уже сохранённые предсказания и метрики доступны в `results/stage_a/` без повторного запуска.

### 5.3. Повторно обучить Stage A (CUDA GPU)

```bash
python -m pip install -r training/requirements-training.txt
python training/train_stage_a.py --data-root /путь/к/Dataset/Dataset --experiment both
```

Исходный эксперимент запускался на NVIDIA Tesla T4 в Colab. Использованы ResNet18 ImageNet-1K V1, `seed=42`, изображения 224×224, batch 32, AdamW, class-weighted CrossEntropy, горизонтальное отражение `p=0.5` и умеренный ColorJitter. Первые две эпохи обучается `fc`, следующие пять — `layer4 + fc`; лучший checkpoint выбирается по Validation macro-F1. Quality и CNN Diversity использовали общую Validation-выборку, однако отличались **и способом отбора, и размером Train**. Повторное обучение может дать несколько отличающиеся числа в другом окружении.

### 5.4. Повторно обучить Stage B (Google Colab с GPU)

`training/train_stage_b_colab.py` — **отдельный Colab-скрипт**, не CPU-тренировщик. В среде Colab с включённым GPU загрузите этот `.py` и выполните в ячейке:

```python
from google.colab import files
from pathlib import Path

uploaded = files.upload()  # выберите training/train_stage_b_colab.py
Path('/content/train_stage_b_colab.py').write_bytes(uploaded['train_stage_b_colab.py'])
%run -i /content/train_stage_b_colab.py
```

Если потребуется, скрипт запросит `split_manifests/stage_a_split_v3.csv`; исходные изображения версии 1 скачает через KaggleHub (или использует уже доступные). Обучение Stage B начинается **заново от ImageNet**, использует все **3 723** качественных фотографии и длится **2 head + 3 fine-tuning эпохи**. Validation и Test отдельно на этой стадии не применяются.

Исследовательский `analysis/dataset_research_code_only.ipynb` содержит 85 кодовых ячеек, но **не является полностью автоматическим `Run all` pipeline**: часть решений по качеству принималась визуально. Зафиксированные списки и подробная техническая история сохранены отдельно.

## 6. Ограничения

- Фотографии исходного датасета сняты не в магазине; возможен **domain shift** из-за фона, камер, освещения, упаковки и перекрытий.
- В разбиении V3 префиксные группы съёмок не пересекаются, но **14 групп «класс + дата» пересекают выборки**. Полная независимость условий съёмки не доказана.
- В Stage A Test только **10** настоящих изображений Cucumber; F1 для Cucumber — **0,5625**. 13 из 16 ошибок Test — Capsicum, предсказанный как Cucumber.
- Для финальной Stage B нет независимого Test после включения прежних Validation/Test в её Train. Её точность на новых магазинных фотографиях **не установлена**.
- Softmax confidence не калибрована; режим `unknown` и ONNX-экспорт не реализованы.

Для интерпретации результатов и обоснования решений см. [REPORT.pdf](REPORT.pdf) и [подробную историю проекта](analysis/RECONCILED_PROJECT_HISTORY.md).
