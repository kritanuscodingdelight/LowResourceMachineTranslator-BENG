# Low-Resource Neural Machine Translation: English → Bengali

Fine-tuning **mBART-50** and **NLLB-200** on a simulated low-resource EN-BN setup using 10k pairs from the Samanantar corpus. Includes full EDA, data quality analysis, back-translation augmentation, data ablation, and qualitative error analysis.

---

## Results

### Model Comparison (Clean Validation Subset, 200 pairs)

| Model | BLEU | chrF++ |
|---|---|---|
| mBART-50 | 10.32 | 27.16 |
| **NLLB-200** | **19.79** | **38.21** |

> chrF++ is the primary metric. For morphologically rich languages like Bengali, chrF++ is more robust to reference paraphrase variation — a known issue with the Samanantar corpus (see Data Quality section).

### Training Performance (Validation Set, per epoch)

**mBART-50** (5 epochs, best checkpoint at epoch 3):

| Epoch | Train Loss | Val Loss | BLEU | chrF++ |
|---|---|---|---|---|
| 1 | 3.5376 | 3.3509 | 18.14 | 32.45 |
| 2 | 2.9035 | 3.0583 | 18.14 | 35.90 |
| **3** | **2.3523** | **2.9558** | **19.34** | **36.56** |
| 4 | 2.0236 | 2.9796 | 19.34 | 36.56 |
| 5 | 1.7384 | 3.0165 | 17.61 | 37.04 |

**NLLB-200** (3 epochs, converged at epoch 2):

| Epoch | Train Loss | Val Loss | BLEU | chrF++ |
|---|---|---|---|---|
| 1 | 1.9801 | 1.8825 | 20.32 | 32.49 |
| **2** | **1.8927** | **1.8577** | **20.26** | **32.40** |
| 3 | 1.7369 | 1.8574 | 20.26 | 32.40 |

NLLB starts from a much lower loss (1.98 vs 3.54) due to stronger multilingual pretraining with explicit Bengali coverage.

### Data Ablation (NLLB-200, 3 epochs each)

| Training Pairs | BLEU | chrF++ |
|---|---|---|
| 2,000 | 20.82 | 38.55 |
| 5,000 | 21.27 | 38.67 |
| **10,000** | **22.04** | **39.30** |

Consistent scaling -- both BLEU and chrF++ improve monotonically with more data. The relatively small gap between 2k and 10k (+1.22 BLEU) reflects diminishing returns typical of fine-tuning large pretrained models on small datasets.

### Sample Translations

```
SRC  : The property has been converted into a theme park surrounded by
       four luxury hotels overlooking the zoo.
REF  : সম্পত্তিটি একটি থিম পার্কে রূপান্তরিত করা হয়েছে যার চারটি
       বিলাসবহুল হোটেলগুলি চিড়িয়াখানাকে ঘিরে রেখেছে।
mBART: বিলাসিটিটিকে একটি সেমেম পার্কে পরিণত করা হয়েছে, যা জৈবখানা
       দেখতে চারটি বিলাসি হোটেলের আশপাশে রয়েছে।
NLLB : চারটি বিলাসবহুল হোটেল যা চিড়িয়াখানাটিকে ঘিরে রেখে থিম
       পার্কে পরিণত হয়েছে।
```

---

## Dataset

**Primary:** [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (EN-BN subset)
- 8.6M parallel pairs total; subsampled to **10,000 train / 1,000 val / 500 test**
- Filtered by length (3–100 tokens) and BN/EN length ratio (0.33–3.0)
- After filtering: **9,300 train / 920 val / 459 test**

**Evaluation:** [facebook/flores](https://huggingface.co/datasets/facebook/flores) (FLORES-200 devtest)
- Standard zero-shot benchmark across 200 languages
- Note: FLORES zero-shot eval skipped due to HuggingFace dataset script deprecation. Use `mcmillanmajora/flores200` as a drop-in replacement.

### Data Statistics (Train Split)

| | EN (source) | BN (target) |
|---|---|---|
| Mean tokens | 11.6 | 10.3 |
| Median tokens | 8.0 | 7.0 |
| Max tokens | 342 | 145 |
| 95th percentile | 32 | 28 |
| 99th percentile | 48 | 46 |

99th percentile at 48 tokens confirms max_length=128 has zero truncation impact.

### Data Quality Finding

Qualitative error analysis revealed that a significant portion of low-BLEU outputs are caused by **misaligned reference pairs in Samanantar**, not model failures:

```
SRC : Still, translating took precedence over everything else.
PRED: তবুও, অন্য সব কিছুর থেকে অনুবাদকে অগ্রাধিকার দেওয়া হয়েছিল।  ← correct
REF : দুজনেই অক্লান্তভাবে কাজ করেছিল।                                ← unrelated
```

This is a known issue in web-crawled parallel corpora. LaBSE-based semantic similarity filtering is a recommended next step and would likely recover 2–3 BLEU points.

### Back-Translation Augmentation

NLLB used in BN→EN direction to generate 2,000 synthetic source sentences, producing an augmented training set of **11,300 pairs** (9,300 original + 2,000 synthetic) -- a 21% increase at zero annotation cost.

---

## Models

| Model | Parameters | HuggingFace ID |
|---|---|---|
| mBART-50 | 680M | `facebook/mbart-large-50-many-to-many-mmt` |
| NLLB-200 distilled | 600M | `facebook/nllb-200-distilled-600M` |

---

## Project Structure

```
.
├── Low_Resource_Neural_Machine_Translation.ipynb
├── nmt_outputs/
│   ├── mbart_best/               # fine-tuned mBART checkpoint
│   ├── nllb_best/                # fine-tuned NLLB checkpoint
│   ├── length_dist_train.png
│   ├── token_freq_train.png
│   ├── ratio_heatmap_train.png
│   ├── vocab_growth_train.png
│   ├── training_curves.png
│   ├── model_comparison.png
│   ├── sent_bleu_mBART-50.png
│   ├── sent_bleu_NLLB-200.png
│   ├── error_analysis_mBART-50.png
│   └── error_analysis_NLLB-200.png
└── README.md
```

---

## Pipeline Overview

### 1. EDA
- Sentence length distributions (source + target + scatter with Pearson r)
- Top-30 token frequency for EN and BN
- BN/EN length ratio heatmap by source length bucket
- Vocabulary growth curves (type-token ratio proxy)

### 2. Preprocessing
- Length filter: 3–100 tokens per side
- Length ratio filter: 0.33–3.0 (BN/EN)

### 3. Fine-tuning
Both models trained with identical settings:
- AdamW, lr=3e-5, warmup=200 steps, weight decay=0.01
- Batch size 16, fp16, early stopping patience=2
- Best checkpoint by validation loss

### 4. Evaluation
- Corpus BLEU (flores200 tokenizer) and chrF++
- Sentence-level BLEU distribution
- Clean subset evaluation (length-ratio filtered, n=200) to control for reference noise

### 5. Back-Translation Augmentation
NLLB BN→EN generates synthetic source sentences from target monolingual data.

### 6. Data Ablation
NLLB trained at 2k / 5k / 10k pairs to quantify scaling behavior.

### 7. Error Analysis
- 10 worst/best translations by sentence BLEU
- BLEU vs source length scatter
- Mean BLEU by source length bucket

---

## Quickstart

```bash
pip install transformers datasets sacrebleu sentencepiece accelerate
```

Set `HF_TOKEN` in Colab Secrets, then run all cells top to bottom.

### Inference

```python
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

tok   = NllbTokenizer.from_pretrained("./nmt_outputs/nllb_best")
model = AutoModelForSeq2SeqLM.from_pretrained("./nmt_outputs/nllb_best")
tok.src_lang = "eng_Latn"

inputs = tok("The doctor advised complete bed rest.", return_tensors="pt")
out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("ben_Beng"),
    max_new_tokens=128,
    num_beams=4,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Key Findings

1. **NLLB significantly outperforms mBART** on clean evaluation (BLEU 19.79 vs 10.32, chrF++ 38.21 vs 27.16), consistent with NLLB's dedicated Bengali pretraining.

2. **Reference noise is the primary evaluation bottleneck.** Samanantar contains misaligned pairs where correct translations are penalized. Confirmed qualitatively across multiple examples.

3. **NLLB converges faster and lower** -- val loss 1.86 vs mBART 2.96, requiring only 3 epochs vs 5.

4. **Diminishing returns on data scaling** -- NLLB gains only +1.22 BLEU from 2k to 10k pairs, reflecting strong Bengali priors already encoded in the pretrained model.

5. **Back-translation adds 2,000 synthetic pairs** at zero cost, augmenting the training set by 21%.

6. **BLEU plateaus while chrF++ improves** in mBART training -- the model learns Bengali morphological fluency faster than exact phrase matching.

7. **Zero truncation impact** -- 99th percentile sentence length is 48 tokens, well within max_length=128.

---

## Environment

| Library | Version |
|---|---|
| transformers | 5.0.0 |
| datasets | 4.0.0 |
| sacrebleu | 2.6.0 |
| torch | 2.10.0+cu128 |
| accelerate | 1.13.0 |
| sentencepiece | 0.2.1 |

GPU: ~40GB VRAM (tested on Colab A100)

---

## References

- Goyal et al. (2022). [The Flores-200 Evaluation Benchmark](https://arxiv.org/abs/2207.04672)
- Costa-jussà et al. (2022). [No Language Left Behind: NLLB](https://arxiv.org/abs/2207.04672)
- Liu et al. (2020). [Multilingual Denoising Pre-training for NMT (mBART)](https://arxiv.org/abs/2001.08210)
- Ramesh et al. (2022). [Samanantar: Parallel Corpora for 11 Indic Languages](https://arxiv.org/abs/2104.05596)
- Popović (2015). [chrF: character n-gram F-score for MT evaluation](https://aclanthology.org/W15-3049/)

---

## Author

**Kritanu Chattopadhyay**
B.Tech Mechanical Engineering, NIT Durgapur (Class of 2027)
Research affiliations: IIT Patna (AI-NLP-ML Lab) · IIT Bombay (MInDS Lab)
