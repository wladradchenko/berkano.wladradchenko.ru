# Руководство по дообучению модели

## Описание

Это руководство описывает, как дообучить предобученную модель `vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all` на новых данных с огородными растениями.

## Формат данных

Модель поддерживает два формата организации данных:

### Вариант 1: Структура папок (рекомендуется)

Организуйте данные следующим образом:

```
data_dir/
├── train/
│   ├── species_id_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── species_id_2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── species_id_1/
    │   ├── image1.jpg
    │   └── ...
    ├── species_id_2/
    │   └── ...
    └── ...
```

**Важно:**

- Названия папок должны соответствовать `species_id` (например, "1355868", "1355869")
- Имена папок в train и val должны совпадать
- Поддерживаемые форматы изображений: `.jpg`, `.jpeg`, `.png`

### Вариант 2: CSV файлы

Создайте два CSV файла: `train.csv` и `val.csv` в директории с данными.

**Формат CSV:**

```csv
image_path,label
/path/to/image1.jpg,1355868
/path/to/image2.jpg,1355869
/path/to/image3.jpg,1355868
...
```

**Колонки:**

- `image_path`: абсолютный или относительный путь к изображению
- `label`: species_id (строка, например "1355868")

**Пример структуры:**

```
data_dir/
├── train.csv
└── val.csv
```

## Предобработка изображений

**ВАЖНО:** Модель автоматически применяет необходимые трансформации. Вам НЕ нужно делать resize или другую предобработку вручную.

### Что делает модель автоматически:

1. **Resize и Crop:**

   - Изображения автоматически приводятся к размеру, требуемому моделью
   - Используется random crop для обучения и center crop для валидации

2. **Нормализация:**

   - Применяется нормализация с параметрами модели

3. **Аугментации (только для обучения):**
   - Random horizontal flip (вероятность 0.5)
   - Color jitter (яркость, контраст, насыщенность)
   - Random scale и ratio

### Требования к исходным изображениям:

- **Формат:** JPG, JPEG или PNG
- **Цвет:** RGB (цветные изображения)
- **Размер:** Любой (будет автоматически изменен)
- **Качество:** Рекомендуется минимум 224x224 пикселей для лучших результатов

### Что НЕ нужно делать:

- ❌ Не нужно делать resize вручную
- ❌ Не нужно применять нормализацию
- ❌ Не нужно конвертировать в grayscale
- ❌ Не нужно делать crop вручную

## Настройка классов

### Вариант 1: Использование существующего class_mapping.txt

Если вы добавляете новые классы к существующим:

1. Откройте `class_mapping.txt`
2. Добавьте новые `species_id` в конец файла (по одному на строку)
3. Убедитесь, что в `species_id_to_name.txt` есть соответствующие записи

**Пример class_mapping.txt:**

```
1355868
1355869
1355870
...
NEW_SPECIES_ID_1
NEW_SPECIES_ID_2
```

### Вариант 2: Автоматическое создание class_mapping.txt

Если вы не указываете `--class_mapping`, скрипт автоматически создаст файл на основе данных в папках train/val.

**Важно:** При автоматическом создании порядок классов определяется алфавитной сортировкой названий папок.

## Обновление species_id_to_name.txt

После добавления новых классов, обновите файл `species_id_to_name.txt`:

**Формат:**

```csv
"species_id";"species"
"1355868";"Taxus baccata L."
"1355869";"Dryopteris filix-mas (L.) Schott"
"NEW_SPECIES_ID_1";"Название нового вида"
```

**Важно:**

- Используйте точку с запятой (`;`) как разделитель
- Используйте кавычки для значений
- species_id должен быть строкой

## Запуск обучения

### Базовый пример:

```bash
python train.py \
    --data_dir /path/to/your/data \
    --pretrained_path /path/to/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar \
    --class_mapping /path/to/class_mapping.txt \
    --output_dir ./outputs \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-5
```

### Пример с автоматическим определением классов:

```bash
python train.py \
    --data_dir /path/to/your/data \
    --pretrained_path /path/to/model_best.pth.tar \
    --output_dir ./outputs \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-5
```

### Пример с дополнительными параметрами:

```bash
python train.py \
    --data_dir /path/to/your/data \
    --pretrained_path /path/to/model_best.pth.tar \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 64 \
    --lr 8e-5 \
    --weight_decay 0.0 \
    --opt adam \
    --sched cosine \
    --warmup_epochs 5 \
    --model_ema \
    --workers 8
```

## Параметры обучения

### Основные параметры:

- `--data_dir`: Путь к директории с данными (обязательно)
- `--pretrained_path`: Путь к предобученной модели (обязательно)
- `--class_mapping`: Путь к class_mapping.txt (опционально)
- `--output_dir`: Директория для сохранения результатов (по умолчанию: ./outputs)

### Гиперпараметры:

- `--epochs`: Количество эпох (по умолчанию: 50)
- `--batch_size`: Размер батча (по умолчанию: 32)
- `--lr`: Начальный learning rate (по умолчанию: 1e-5)
- `--weight_decay`: Weight decay (по умолчанию: 0.0)
- `--opt`: Оптимизатор: adam, sgd (по умолчанию: adam)
- `--sched`: Планировщик LR: cosine, step (по умолчанию: cosine)
- `--label_smoothing`: Label smoothing (по умолчанию: 0.1)

### Рекомендуемые значения:

Для дообучения на новых данных рекомендуется:

- **Learning rate:** 1e-5 до 1e-4 (меньше чем для обучения с нуля)
- **Batch size:** 32-64 (зависит от размера GPU)
- **Epochs:** 30-100 (зависит от размера датасета)
- **Weight decay:** 0.0 (как в оригинальной модели)

## Результаты обучения

После обучения в `output_dir` будут созданы:

1. **model_best.pth.tar** - лучшая модель (по точности на валидации)
2. **last.pth.tar** - последний чекпоинт
3. **class_mapping.txt** - маппинг классов (если создан автоматически)
4. **training_history.csv** - история обучения (loss, accuracy по эпохам)

## Использование обученной модели

После обучения используйте модель так же, как оригинальную:

```python
import timm
import torch
from PIL import Image

# Загрузите модель
model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=False,
    num_classes=YOUR_NUM_CLASSES,
    checkpoint_path='./outputs/model_best.pth.tar'
)
model.eval()

# Загрузите изображение и примените трансформации
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

image = Image.open('your_image.jpg')
image = transforms(image).unsqueeze(0)

# Предсказание
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
```

## Часто задаваемые вопросы

### Q: Нужно ли делать resize изображений перед обучением?

**A:** Нет, модель автоматически применяет все необходимые трансформации.

### Q: Можно ли использовать изображения разных размеров?

**A:** Да, модель автоматически изменяет размер всех изображений.

### Q: Как добавить новые классы к существующим?

**A:** Добавьте новые species_id в конец class_mapping.txt и убедитесь, что они есть в species_id_to_name.txt.

### Q: Что делать, если у меня мало данных?

**A:** Используйте меньший learning rate (1e-5), больше эпох, и включите аугментации (они включены по умолчанию).

### Q: Можно ли использовать модель без дообучения?

**A:** Да, используйте basic_usage_pretrained_model.py с оригинальной моделью.

## Требования

- Python 3.7+
- PyTorch 2.2.1+
- timm 0.9.16+
- pandas
- Pillow
- tqdm

## Примечания

- Модель использует архитектуру ViT-Base с patch size 14
- Предобучена на DINOv2 (self-supervised learning)
- Исходная модель обучена на 7806 классах PlantNet
- При добавлении новых классов классификатор будет переинициализирован
