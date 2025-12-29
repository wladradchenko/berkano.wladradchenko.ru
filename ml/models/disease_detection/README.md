---
license: cc-by-4.0
language:
- en
metrics:
- accuracy
- recall
pipeline_tag: image-to-text
tags:
- agriculture
- leaf
- disease
datasets:
- enalis/LeafNet
library_name: transformers
---

# 🌿 SCOLD: A Vision-Language Foundation Model for Leaf Disease Identification

**SCOLD** is a multimodal model that maps **images** and **text descriptions** into a shared embedding space. This model is developed for **cross-modal retrieval**, **few-shot classification**, and **explainable AI in agriculture**, especially for plant disease diagnosis from both images and domain-specific text prompts.

---

### ✅ Intended Use
- Vision-language embedding for classification or retrieval tasks
- Few-shot learning in agricultural or medical datasets
- Multimodal interpretability or zero-shot transfer
---

## 🧪 How to Use

First clone our repository:

```bash
 git clone https://huggingface.co/enalis/scold
```

Please find detail to load and use our model in *inference.py*

```python

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text = "A maize leaf with bacterial blight"
inputs = tokenizer(text, return_tensors="pt")

# Image preprocessing
image = Image.open("path_to_leaf.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    image_emb, text_emb = model(image_tensor, inputs["input_ids"], inputs["attention_mask"])
    similarity = torch.nn.functional.cosine_similarity(image_emb, text_emb)
    print(f"Similarity score: {similarity.item():.4f}")
```
Please cite this paper if this code is useful for you!

```
@article{NGUYENQUOC2025130084,
title = {A Vision-Language Foundation Model for Leaf Disease Identification},
journal = {Expert Systems with Applications},
pages = {130084},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.130084},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425037005},
author = {Khang {Nguyen Quoc} and Lan Le {Thi Thu} and Luyl-Da Quach},
keywords = {Leaf disease identification, Contrastive learning, Vision-language models, Foundation models, Image-text retrieval, Context-aware learning},
abstract = {Leaf disease identification plays a pivotal role in smart agriculture. However, many existing studies still struggle to integrate image and textual modalities to compensate for each other’s limitations. Furthermore, many of these approaches rely on pretraining with constrained datasets such as ImageNet, which lack domain-specific information. The research proposes SCOLD (Soft-target COntrastive learning for Leaf Disease identification), a context-aware vision-language foundation model tailored to domain-specific tasks in smart agriculture. SCOLD is developed using a diverse corpus of plant leaf images and corresponding symptom descriptions, comprising over 186,000 image-captions pairs aligned with 97 unique concepts. Through task-agnostic pretraining, SCOLD leverages contextual soft targets to mitigate overconfidence in contrastive learning by smoothing labels, thereby improving model generalization and robustness on fine-grained classification tasks. Experimental results demonstrate that SCOLD outperforms existing Vision-language models (VLMs) such as LLaVA 1.5, Qwen-VL 2.5, OpenAI-CLIP-L, BioCLIP, and SigLIP2 across several benchmarks, including zero-shot and few-shot classification, image-text retrieval, and image classification, while maintaining a competitive parameter footprint. Ablation studies further highlight SCOLD’s effectiveness in contrast to its counterparts. The proposed approach significantly advances the agricultural vision-language foundation model, offering strong performance with minimal or no supervised fine-tuning. This work lays a solid groundwork for future research on models trained with long-form and simplified contexts, tasks involving class ambiguity, and multi-modal systems for intelligent plant disease diagnostics. The code for this study is available at https://huggingface.co/enalis/scold.}
}

```