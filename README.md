# 🌐 Comparison of Monolingual and Multilingual BERT Models for Named Entity Recognition (NER) in Nepali

## 🔬 Project Overview
This repository explores the effectiveness of **monolingual BERT** (NepBERTa) vs. **multilingual BERT** (mBERT) for **Named Entity Recognition (NER) in Nepali**.

With low-resource languages like Nepali, multilingual models often under-perform due to **limited representation** in their pre-training corpus. We investigate whether fine-tuning a **monolingual BERT model** yields better results for NER.

## 📚 Dataset
We used a labeled Nepali NER dataset [**EverestNER**](https://github.com/nowalab/everest-ner) containing **PERSON, LOCATION, ORGANIZATION, EVENTS AND DATES** entity types. Preprocessing involves:
- Tokenization using **WordPiece (BERT) tokenizer**
- Converting labels into BIO format
- Splitting into train/validation/test sets

## 🤖 Model Architectures
### 📖 Monolingual: (**NepBERTa**)[https://huggingface.co/NepBERTa/NepBERTa]
- Trained exclusively on **NepaliNER**[https://github.com/dadelani/nepali-ner] dataset
- Stronger linguistic alignment with Nepali syntax & morphology

### 🌐 Multilingual: mBERT
- Trained on **over 100 languages**
- Cross-lingual generalization but weaker specialization for Nepali

## 💡 Experimental Setup
- **Fine-tuned both models** on the Nepali NER dataset using **Hugging Face Transformers**
- Evaluation metrics: **F1-score, Precision, Recall**
- Training setup:
  - **Optimizer**: AdamW
  - **Batch Size**: 32
  - **Epochs**: 5
  - **Learning Rate**: 2e-5

## 📊 Results & Findings
| Model       | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| mBERT      | 87.45%     | 86.08%  | 86.76%    |
| NepBERTa   | 89.65%     | 87.7%  | **88.67%** |

### ⚡ Key Takeaways:
- **NepBERTa significantly outperforms mBERT** in all metrics
- **Multilingual BERT struggles** with Nepali-specific grammar and tokenization
- **Domain-specific training on Nepali data** improves contextual understanding

## 🛠️ Installation & Usage
Clone the repository and install dependencies:
```bash
git clone https://github.com/karkidilochan/Nepali-NER-BERT.git
cd Nepali-NER-BERT
pip install -r requirements.txt
```

## 💬 Future Work
- Leverage **multi-GPU environments** to bridge the gap toward NepBERTa’s reported accuracy.
- Experiment with **LoRA (Low-Rank Adaptation) and Adapter-based fine-tuning** to achieve parameter-efficient model updates.
- **Data augmentation** for Nepali NER
- Fine-tune other models like **Llama-2** for comparison

---
**Contributions & Feedback:** PRs and discussions are welcome! 🚀

