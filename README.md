[![Price](https://img.shields.io/badge/price-FREE-0098f7.svg)](https://github.com/wladradchenko/berkano.wladradchenko.ru/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![GitHub package version](https://img.shields.io/github/v/release/wladradchenko/berkano.wladradchenko.ru?display_name=tag&sort=semver)](https://github.com/wladradchenko/berkano.wladradchenko.ru)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-blue.svg)](https://github.com/wladradchenko/berkano.wladradchenko.ru/blob/main/LICENSE)
<br>
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/wladradchenko/berkano.wladradchenko.ru)
[![Patreon Support](https://img.shields.io/badge/Patreon-Support-white?style=flat&logo=patreon&logoColor=white)](https://patreon.com/wladradchenko)

<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/berkano.wladradchenko.ru">
    <img src="https://github.com/wladradchenko/berkano.wladradchenko.ru/blob/main/assets/leaves.svg" width="200px" height="200px">
  </a>

  <h3 align="center">Berkano</h3>

  <p align="center">
    <a href="https://github.com/wladradchenko/berkano.wladradchenko.ru/wiki">Documentation</a>
    <br/>
    <a href="https://github.com/wladradchenko/berkano.wladradchenko.ru/issues">Issue</a>
    ·
    <a href="https://github.com/wladradchenko/berkano.wladradchenko.ru/discussions">Discussions</a>
  </p>
</div>

# Plant Analysis Project

This repository contains a complete project for **plant analysis using machine learning**, including both **Python-based ML models** and a **mobile application** for on-device inference.

---

## Project Structure

```
.
├── ml/                  # Python ML code
│   ├── models/          # Code for training and inference
│   │   ├── disease_detection/
│   │   ├── plant_analysis/
│   │   └── plant_classification/
│   ├── quantization/    # Scripts for ONNX model quantization
│   └── requirements.txt # Python dependencies
│
└── mobile/              # React Native mobile app
    ├── assets/          # ML models and data files
    ├── android/         # Android native code
    ├── ios/             # iOS native code
    ├── src/             # React Native JS/TS code
    ├── package.json     # Node dependencies
    └── ...              # Files of React Native code
```

---

## ML Models (Python)

The `ml` folder contains all code for:

1. **Training new models** on garden plant datasets.
2. **Inference scripts** for:

   * Plant disease detection
   * Plant species classification
   * Plant age and leaf count estimation

It includes scripts for:

* Loading ONNX or PyTorch models
* Preprocessing images
* Predicting outputs
* Quantizing ONNX models for mobile optimization

Models can be downloaded from:
[https://huggingface.co/wladradchenko/berkano.wladradchenko.ru](https://huggingface.co/wladradchenko/berkano.wladradchenko.ru)

---

## Mobile Application

The `mobile` folder contains a **React Native (vanilla, no Expo) app** that demonstrates running the ML models on a mobile device.

* Users can upload images of plants for:

  * **Species classification**
  * **Age & leaf count estimation**
  * **Disease detection** (from leaves, stems, flowers)
* Uses ONNX Runtime for mobile inference
* Compatible with **Android** and **iOS**
* Requires models and supporting files to be placed in `mobile/assets` (from the same HuggingFace repository)

For detailed instructions on running the mobile app, see the `mobile/README.md`.

---

## License

This project is licensed under the **Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.
You are free to **share** the project, **but not use it commercially or modify it**.

Full license details: [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## References

* ML models repository: [https://huggingface.co/wladradchenko/berkano.wladradchenko.ru](https://huggingface.co/wladradchenko/berkano.wladradchenko.ru)
* React Native app demonstrates **mobile adaptation** of these models.

### Supporters and Donors

You can support the author of the project in the development of his creative ideas on <a href="https://www.patreon.com/c/wladradchenko">Patreon</a> or <a href="https://wladradchenko.ru/donat">CloudTips</a>. 

<!-- CONTACT -->
## Contact

Owner: [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project on GitHub: [https://github.com/wladradchenko/berkano.wladradchenko.ru](https://github.com/wladradchenko/berkano.wladradchenko.ru)

<p align="right">(<a href="#top">to top</a>)</p>

