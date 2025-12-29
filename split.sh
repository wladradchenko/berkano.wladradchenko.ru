#!/bin/bash

# Корневая папка с категориями
SRC_DIR="data"
TRAIN_DIR="$SRC_DIR/train"
VAL_DIR="$SRC_DIR/val"

# Создаем папки train и val, если их нет
mkdir -p "$TRAIN_DIR"
mkdir -p "$VAL_DIR"

# Проходим по всем папкам с изображениями
for category in "$SRC_DIR"/*; do
    if [ -d "$category" ]; then
        folder_name=$(basename "$category")
        echo "Обрабатываем категорию: $folder_name"

        # Создаем соответствующие папки в train и val
        mkdir -p "$TRAIN_DIR/$folder_name"
        mkdir -p "$VAL_DIR/$folder_name"

        # Список всех файлов в категории
        files=("$category"/*)
        total=${#files[@]}

        # Количество файлов для train (85%)
        train_count=$(( total * 85 / 100 ))

        # Перемешиваем файлы
        shuffled=($(shuf -e "${files[@]}"))

        # Копируем в train
        for file in "${shuffled[@]:0:train_count}"; do
            cp "$file" "$TRAIN_DIR/$folder_name/"
        done

        # Копируем в val
        for file in "${shuffled[@]:train_count}"; do
            cp "$file" "$VAL_DIR/$folder_name/"
        done
    fi
done

echo "Разделение данных завершено!"
