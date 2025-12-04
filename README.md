# Clinical Discharge Summarization using MedGemma 4B with QLoRA

A Gen AI semester project focused on fine-tuning large language models for generating comprehensive patient hospital
discharge summaries using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Overview

This project demonstrates fine-tuning the MedGemma 4B model using QLoRA (Quantized Low-Rank Adaptation) to generate
detailed clinical discharge summaries from patient notes. The goal is to achieve high recall by capturing all critical
medical entities including diagnoses, medications, vitals, lab results, procedures, and follow-up instructions.

## Key Technologies

- **Model**: google/medgemma-4b-it (or base Gemma-4b)
- **Fine-tuning Technique**: QLoRA (4-bit quantization with LoRA adapters)
- **Evaluation**: Clinical BERTScore using Bio_ClinicalBERT
- **Platform**: Google Colab / Consumer GPUs with limited VRAM

## Features

- Memory-efficient training using 4-bit quantization
- LoRA adapters for parameter-efficient fine-tuning
- Clinical-specific evaluation metrics (BERTScore with Bio_ClinicalBERT)
- Support for both local and Google Colab execution
- Comprehensive evaluation and qualitative analysis

## Setup

### Requirements

Python 3.12+ with the following dependencies (managed via `uv`):

```bash
# Install dependencies
uv sync
```

### Running the Notebooks

#### 1. Data Cleaning (`data_cleaning.ipynb`)

Run this notebook first to prepare the dataset:

1. Download the MIMIC-IV dataset from Kaggle (requires Kaggle API credentials)
2. Execute all cells sequentially to perform data preprocessing
3. Outputs will be saved as CSV files in the project directory

**Note**: Download NLTK data when prompted:

```python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

#### 2. Model Fine-tuning (`clinical_discharge_summarization_peft.ipynb`)

After data cleaning, run the training notebook:

- **Local execution**: Run cells starting from Section 3B (Load The Dataset)
- **Google Colab**: Run Section 2A to mount Drive, then Section 3A (Colab)

#### 3. Two-Stage Compression Pipeline (`two_stage_compression_pipeline.ipynb`)

After fine-tuning, run the compression pipeline:

- **Automatic environment detection**: Detects Colab vs Local automatically
- **Local execution**: Uses local file paths, saves to `./compression_results/`
- **Google Colab**:
    1. Mounts Google Drive automatically
    2. Update paths in Section 2 to match your Drive structure
    3. Requires HuggingFace authentication for Llama-3

**Important**: This notebook manages memory aggressively to run two large models sequentially on a single GPU.

## Project Structure

```
.
├── data_cleaning.ipynb                           # Data preprocessing pipeline
├── clinical_discharge_summarization_peft.ipynb  # Stage 1: MedGemma fine-tuning
├── two_stage_compression_pipeline.ipynb          # Stage 2: Llama compression pipeline
├── pyproject.toml                                # Project dependencies
├── uv.lock                                       # Locked dependencies
└── README.md                                     # This file
```

## Notebooks

### 1. Data Cleaning Pipeline (`data_cleaning.ipynb`)

This notebook performs comprehensive data preprocessing on the MIMIC-IV BHC dataset:

**Data Loading**:

- Downloads MIMIC-IV dataset from Kaggle using `kagglehub`
- Dataset: `aminexdr/bhc-mimic-iv-summary` (269,516 clinical note entries)

**Data Quality Checks**:

- Missing value detection and handling
- Duplicate row removal
- Datetime conversion for `charttime` and `storetime` columns

**Text Preprocessing Pipeline**:

1. **Text Cleaning**:
    - Lowercasing all text
    - Removing numbers and special characters
    - Removing excess whitespace

2. **Tokenization**:
    - Using NLTK's `word_tokenize` for splitting text into tokens
    - Creates `tokenized_input` and `tokenized_target` columns

3. **Stopword Removal & Lemmatization**:
    - Removes common English stopwords using NLTK's stopwords corpus
    - Applies WordNet lemmatization to reduce words to base forms
    - Creates `lemmatized_input` and `lemmatized_target` columns

4. **Rare Word Filtering**:
    - Identifies and removes words appearing ≤2 times
    - Creates `filtered_input` column

5. **Text Reconstruction**:
    - Joins tokens back into strings
    - Creates `final_input` and `final_target` columns for model training

**Visualizations**:

- Input/target length distributions using seaborn histplots
- Statistical analysis of text lengths

**Output Files**:

- `mimic_cleaned_full.csv`: Complete dataset with all preprocessing steps
- `mimic_cleaned_text_only.csv`: Final input/target pairs for training
- `mimic_preprocessing_steps.csv`: Intermediate preprocessing stages for analysis

**Required Libraries**:

- kagglehub, pandas, numpy, nltk, matplotlib, seaborn, re, collections

### 2. Model Fine-tuning (`clinical_discharge_summarization_peft.ipynb`)

This notebook fine-tunes MedGemma 4B for clinical discharge summarization using QLoRA:

**Environment Setup**:

- Installs transformers, peft, bitsandbytes, trl, accelerate, datasets, bert_score
- Configures GPU/CUDA environment
- Supports both local and Google Colab execution

**Data Preparation**:

- Loads `mimic_cleaned_text_only.csv` from data cleaning pipeline
- Splits into train/test sets (95/5 split)
- Formats data using Gemma's conversation template with special tokens
- Creates instruction-based prompts emphasizing complete medical entity coverage

**QLoRA Configuration**:

1. **4-bit Quantization**:
    - NF4 (4-bit NormalFloat) quantization type
    - Double quantization for additional memory savings
    - Float16 compute dtype for numerical stability

2. **LoRA Adapters**:
    - Rank: 32, Alpha: 64
    - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Dropout: 0.05
    - Trains only ~0.5% of total parameters

**Training Process**:

- Uses SFTTrainer from TRL library
- Gradient accumulation (8 steps) for effective batch size
- AdamW optimizer with cosine learning rate schedule
- Gradient checkpointing for memory efficiency
- Warmup steps: 100, Learning rate: 2e-4

**Evaluation**:

- Clinical BERTScore using Bio_ClinicalBERT embeddings
- Measures Precision, Recall, and F1 scores
- Focuses on recall to ensure comprehensive medical entity capture
- Qualitative analysis of generated vs reference summaries

**Generation Parameters**:

- Max new tokens: 512 for detailed summaries
- Temperature: 0.7 for balanced creativity
- Top-p: 0.9, Top-k: 50 for nucleus sampling
- Repetition penalty: 1.1

**Model Outputs**:

- Trained LoRA adapter weights (`adapter_model.bin`)
- Configuration files (`adapter_config.json`)
- Tokenizer files for inference
- Checkpoint files for model recovery

**Required Libraries**:

- torch, transformers, peft, bitsandbytes, trl, accelerate, datasets, bert_score, scipy, einops

### 3. Two-Stage Compression Pipeline (`two_stage_compression_pipeline.ipynb`)

This notebook implements a two-stage clinical summarization pipeline with compression:

**Pipeline Architecture**:

```
Clinical Notes → [Stage 1: MedGemma-4B] → Verbose Summary → [Stage 2: Llama-3-8B] → Compressed Summary
```

**Stage 1: Verbose Summary Generation**:

1. **Model Loading**:
    - Loads fine-tuned MedGemma-4B with LoRA adapters
    - 4-bit quantization (NF4) for memory efficiency
    - Generates high-recall, detailed summaries

2. **Generation Parameters**:
    - Max tokens: 512 for comprehensive coverage
    - Temperature: 0.7 for balanced outputs
    - Emphasizes medical entity completeness

3. **Critical Memory Management**:
    - Complete model unloading after generation
    - `flush_memory()` utility for GPU cache clearing
    - Ensures no OOM errors on single GPU

**Stage 2: Compression with Entity Retention**:

1. **Model Loading**:
    - Loads Meta-Llama-3-8B-Instruct in 4-bit
    - Only loaded after MedGemma is fully unloaded

2. **Chain-of-Density Inspired Compression**:
    - Prompt: "Rewrite to be 50% shorter while retaining ALL entities"
    - Preserves: medications (with dosages), vitals (with numbers), lab results, diagnoses, procedures, dates
    - Temperature: 0.3 for deterministic compression

**Evaluation Metrics** (Research Component):

1. **Compression Ratio**:
    - Measures: `len(Stage2) / len(Stage1)`
    - Target: ~50% compression
    - Reports: mean, median, std dev

2. **Entity Retention (NER)**:
    - Uses SciSpacy (`en_core_sci_sm`) for medical entity extraction
    - Calculates recall: `entities_in_stage2 / entities_in_stage1`
    - Critical metric: Should be >85% for clinical use

3. **Clinical BERTScore**:
    - Compares compressed summaries vs original notes
    - Uses `Bio_ClinicalBERT` embeddings
    - Measures semantic similarity (Precision, Recall, F1)

**Memory Management Architecture**:

- `flush_memory()`: Clears GPU cache using `torch.cuda.empty_cache()` + `gc.collect()`
- `unload_model()`: Moves model to CPU, deletes references, flushes memory
- Sequential loading: MedGemma → unload → Llama (prevents OOM)

**Output Files**:

- `compression_pipeline_results.csv`: Full results with all metrics
- `compression_pipeline_results.json`: Structured results for analysis
- `summary_statistics.json`: Aggregate metrics and configuration

**Environment Compatibility**:

- **Auto-detection**: Automatically detects Colab vs Local execution
- **Local Mode**: Uses project directory paths
- **Colab Mode**:
    - Automatic Google Drive mounting
    - Configurable Drive paths for inputs/outputs
    - HuggingFace authentication for Llama-3

**Analysis Features**:

- Correlation analysis (compression vs entity retention)
- Best/worst sample identification
- Sample-by-sample comparison tables

**Required Libraries**:

- torch, transformers, bitsandbytes, accelerate, scispacy, bert_score, pandas, numpy

## Model Configuration

- **LoRA Rank**: 32
- **Training**: 1 epoch with batch size 1 and gradient accumulation
- **Max Sequence Length**: 2048 tokens
- **Generation**: Up to 512 new tokens with temperature 0.7

## Evaluation

The model is evaluated using Clinical BERTScore, which measures semantic similarity using Bio_ClinicalBERT embeddings.
This approach is superior to traditional metrics (BLEU/ROUGE) for medical text as it captures clinical context and
medical terminology.

## Usage

After training, use the inference function to generate summaries for new clinical notes:

```python
summary = generate_discharge_summary(clinical_notes)
print(summary)
```

## Dataset

The project uses the MIMIC-IV BHC clinical discharge summaries dataset:

**Source**: [MIMIC-IV BHC Summary Dataset](https://www.kaggle.com/datasets/aminexdr/bhc-mimic-iv-summary) (Kaggle)

- 269,516 clinical note entries
- Contains discharge summaries from MIMIC-IV clinical database

**Data Pipeline**:

1. **Raw Dataset** (`data_cleaning.ipynb`):
    - `input`: Raw clinical notes with "summarize:" prefix
    - `target`: Reference discharge summaries
    - Additional metadata: note_id, subject_id, hadm_id, timestamps

2. **Preprocessed Dataset** (`mimic_cleaned_text_only.csv`):
    - `final_input`: Cleaned, tokenized, and lemmatized clinical notes
    - `final_target`: Cleaned, tokenized, and lemmatized summaries
    - Ready for model training

**Preprocessing Steps**:

- Text cleaning (lowercasing, removing numbers/special characters)
- Tokenization using NLTK
- Stopword removal
- Lemmatization using WordNet
- Rare word filtering (≤2 occurrences)

## Collaborators

- [Bryan Bergo](https://github.com/blbergo)
- [Mike Leon](https://github.com/mikeleon001)
- [Nicolas Escobedo](https://github.com/Nico25041)
- [Trung Vong](https://github.com/TrungVN9)
- [Talha Ahmed](https://github.com/talhaahmed884)

## License

This is an academic project for educational purposes.