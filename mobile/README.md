# Plant Analysis Mobile App

This folder contains a **React Native** project (without Expo) demonstrating mobile usage of the plant ML models for **disease detection**, **plant classification**, and **age & leaf count estimation**.

The app allows users to:

1. Upload a photo of the whole plant for **species classification** and **age/leaf count estimation**.
2. Upload specific photos of leaves, stems, or other plant parts for **disease detection**.

It serves as an example of **adapting ML models for mobile devices** using ONNX and React Native.

---

## Project Structure

```
mobile/
├── android/               # Android native code
├── ios/                   # iOS native code
├── assets/                # ML models and data files
│   ├── models/            # ONNX models
│   └── files/             # embeddings, captions, mappings
├── src/                   # React Native JavaScript/TypeScript code
├── package.json           # Node dependencies
└── ...
```

---

## Setup

### 1. Download Models and Assets

Clone the repository with ML models:

```bash
git clone https://huggingface.co/wladradchenko/berkano.wladradchenko.ru
```

Copy the following into the `mobile/assets` folder:

* `models/` → contains `disease_detection.onnx`, `plant_classification.onnx`, `plant_analysis.onnx`
* `files/` → contains `embeddings.bin`, `captions.json`, `class_mapping.txt`, `species_id_to_name.txt`, etc.

---

### 2. Install Dependencies

Inside the `mobile` folder, install Node.js dependencies:

```bash
npm install
```

This will create the `node_modules/` folder.

---

### 3. Start the Metro Bundler

Start the React Native development server:

```bash
npm start
```

Keep this terminal open while running the app.

---

### 4. Run on Android

1. Launch your Android emulator, e.g.:

```bash
emulator -avd Pixel_8_Pro
```

Tested successfully on **Pixel 8 Pro**.

2. In a separate terminal, run the Android app:

```bash
npm run android
```

The app will build and deploy to the emulator.

---

### 5. Run on iOS

1. Ensure you have **Xcode** and an iOS simulator installed.
2. Install CocoaPods dependencies:

```bash
cd ios && pod install && cd ..
```

3. Start the iOS app:

```bash
npx react-native run-ios
```

Select the desired simulator from Xcode if needed.

---

## How It Works

The mobile app interacts with the three ML models:

1. **Disease Detection** – Users upload images of leaves, stems, or flowers. The app converts the image into a feature vector and compares it against `embeddings.bin` to detect plant diseases.
2. **Plant Classification** – Users upload an image of the whole plant. The model predicts the species based on the ONNX classifier and `class_mapping.txt` → `species_id_to_name.txt`.
3. **Age & Leaf Count Estimation** – Users upload the full plant image. The MVVT model estimates the plant's age (in days) and leaf count.

All models are optimized for mobile use via ONNX.

---

## Notes

* The app uses **ONNX Runtime** for mobile inference.
* The `assets` folder must contain **all required models and mapping files**.
* The app demonstrates **uploading photos**, preprocessing them, running inference, and displaying results in a mobile-friendly UI.
* Works both on **Android** and **iOS**, with hardware acceleration when available.

---

This setup provides a **complete mobile demonstration** of plant ML models and can be extended for custom datasets, additional plant species, or integration into production mobile apps.

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>

