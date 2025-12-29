# Image Search App with ONNX Runtime Mobile

Мобильное приложение на React Native для поиска похожих изображений с использованием ONNX Runtime Mobile. Приложение использует GPU/NPU через NNAPI для выполнения модели машинного обучения на устройстве.

## Особенности

- ✅ ONNX Runtime Mobile с поддержкой GPU/NPU через NNAPI
- ✅ Поиск по векторным представлениям (embeddings) с использованием косинусного расстояния
- ✅ Настраиваемое количество результатов поиска (топ-K)
- ✅ Загрузка изображений из галереи
- ✅ Чистый React Native (без Expo) для максимальной производительности

## Требования

- React Native 0.83.1+
- Node.js >= 20
- Android SDK (для Android)
- Xcode (для iOS)
- Минимальные требования устройства: Tensor G2 или эквивалент/мощнее
- Android 8.1+ (для NNAPI поддержки)

## Структура проекта

```
src/
├── assets/
│   ├── db/
│   │   ├── embeddings.bin    # Векторные представления (Float32Array)
│   │   └── captions.json    # Метки для каждого вектора
│   ├── images/
│   │   └── image.jpg        # Тестовое изображение
│   └── models/
│       └── scold_image_only_fp16.onnx  # Квантизированная ONNX модель
├── components/
│   └── ImageSearchScreen.tsx  # Главный UI компонент
└── services/
    ├── imagePreprocessing.ts  # Предобработка изображений
    ├── onnxService.ts         # Работа с ONNX моделью
    └── embeddingsService.ts   # Загрузка embeddings и поиск
```

## Установка зависимостей

```bash
npm install
```

## Важные замечания

### Предобработка изображений

⚠️ **Текущая реализация предобработки изображений упрощенная**. Для production необходимо создать нативный модуль для декодирования изображений в тензор формата `[1, 3, 224, 224]` с нормализацией значений в диапазоне `[0, 1]`.

Текущий код использует заглушку для тестирования. Для реальной работы нужно:

1. Создать нативный модуль (Android/iOS) для декодирования изображений
2. Или использовать библиотеку типа `react-native-fast-image` с нативным расширением

### Загрузка assets

Файлы `embeddings.bin` и `captions.json` копируются из assets в локальную файловую систему при первом запуске. Убедитесь, что файлы находятся в правильных директориях.

### ONNX Runtime

Модель автоматически использует NNAPI (GPU/NPU) если доступно, иначе fallback на CPU. Execution providers настраиваются в `src/services/onnxService.ts`.

## Запуск приложения

### Android

```bash
npm run android
```

### iOS

Сначала установите CocoaPods зависимости:

```bash
bundle install
bundle exec pod install
```

Затем запустите:

```bash
npm run ios
```

## Использование

1. Нажмите "Выбрать изображение" для выбора изображения из галереи
2. Настройте количество результатов (по умолчанию 10)
3. Нажмите "Найти похожие" для выполнения поиска
4. Результаты отображаются с расстоянием и меткой (caption)

## Производительность

- Модель загружается при старте приложения
- **Embeddings и captions загружаются в нативную память (не в JavaScript heap)** - решает проблему OOM для больших файлов (>200MB)
- Поиск выполняется в нативной памяти через нативный модуль `FaissSearch`
- Данные не попадают в JavaScript heap, что позволяет работать с большими базами данных
- Поиск выполняется с вычислением косинусного расстояния для всех векторов (линейный поиск O(n))

## Известные ограничения

1. **Предобработка изображений**: ✅ Реализовано через нативный модуль `ImageDecoder`
2. **Поиск по векторам**: ✅ Реализовано через нативный модуль `FaissSearch` (данные в нативной памяти)
3. **Загрузка assets**: Для iOS может потребоваться дополнительная настройка
4. **Поиск**: Текущая реализация использует линейный поиск O(n). Для очень больших баз (>1M векторов) можно добавить индексацию через Faiss IndexFlat или IndexIVF

## Дальнейшие улучшения

- [x] Создать нативный модуль для предобработки изображений
- [x] Интегрировать нативный модуль для работы с большими файлами (данные в нативной памяти)
- [ ] Добавить индексацию через Faiss IndexFlat/IndexIVF для ускорения поиска на очень больших базах
- [ ] Оптимизировать парсинг JSON для очень больших файлов captions
- [ ] Добавить поддержку камеры для захвата изображений
- [ ] Добавить кэширование результатов поиска

---

This is a [**React Native**](https://reactnative.dev) project, bootstrapped using [`@react-native-community/cli`](https://github.com/react-native-community/cli).

# Getting Started

> **Note**: Make sure you have completed the [Set Up Your Environment](https://reactnative.dev/docs/set-up-your-environment) guide before proceeding.

## Step 1: Start Metro

First, you will need to run **Metro**, the JavaScript build tool for React Native.

To start the Metro dev server, run the following command from the root of your React Native project:

```sh
# Using npm
npm start

# OR using Yarn
yarn start
```

## Step 2: Build and run your app

With Metro running, open a new terminal window/pane from the root of your React Native project, and use one of the following commands to build and run your Android or iOS app:

### Android

```sh
# Using npm
npm run android --deviceId Pixel_8_Pro or before this do emulator -avd Pixel_8_Pro and after npm run android

# OR using Yarn
yarn android
```

### iOS

For iOS, remember to install CocoaPods dependencies (this only needs to be run on first clone or after updating native deps).

The first time you create a new project, run the Ruby bundler to install CocoaPods itself:

```sh
bundle install
```

Then, and every time you update your native dependencies, run:

```sh
bundle exec pod install
```

For more information, please visit [CocoaPods Getting Started guide](https://guides.cocoapods.org/using/getting-started.html).

```sh
# Using npm
npm run ios

# OR using Yarn
yarn ios
```

If everything is set up correctly, you should see your new app running in the Android Emulator, iOS Simulator, or your connected device.

This is one way to run your app — you can also build it directly from Android Studio or Xcode.

## Step 3: Modify your app

Now that you have successfully run the app, let's make changes!

Open `App.tsx` in your text editor of choice and make some changes. When you save, your app will automatically update and reflect these changes — this is powered by [Fast Refresh](https://reactnative.dev/docs/fast-refresh).

When you want to forcefully reload, for example to reset the state of your app, you can perform a full reload:

- **Android**: Press the <kbd>R</kbd> key twice or select **"Reload"** from the **Dev Menu**, accessed via <kbd>Ctrl</kbd> + <kbd>M</kbd> (Windows/Linux) or <kbd>Cmd ⌘</kbd> + <kbd>M</kbd> (macOS).
- **iOS**: Press <kbd>R</kbd> in iOS Simulator.

## Congratulations! :tada:

You've successfully run and modified your React Native App. :partying_face:

### Now what?

- If you want to add this new React Native code to an existing application, check out the [Integration guide](https://reactnative.dev/docs/integration-with-existing-apps).
- If you're curious to learn more about React Native, check out the [docs](https://reactnative.dev/docs/getting-started).

# Troubleshooting

If you're having issues getting the above steps to work, see the [Troubleshooting](https://reactnative.dev/docs/troubleshooting) page.

# Learn More

To learn more about React Native, take a look at the following resources:

- [React Native Website](https://reactnative.dev) - learn more about React Native.
- [Getting Started](https://reactnative.dev/docs/environment-setup) - an **overview** of React Native and how setup your environment.
- [Learn the Basics](https://reactnative.dev/docs/getting-started) - a **guided tour** of the React Native **basics**.
- [Blog](https://reactnative.dev/blog) - read the latest official React Native **Blog** posts.
- [`@facebook/react-native`](https://github.com/facebook/react-native) - the Open Source; GitHub **repository** for React Native.
