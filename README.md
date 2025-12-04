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

- **Memory-Efficient Training**: 4-bit quantization enables training 4B models on consumer GPUs
- **Parameter-Efficient Fine-Tuning**: LoRA adapters train only ~0.5% of parameters
- **Clinical-Specific Evaluation**: BERTScore with Bio_ClinicalBERT embeddings
- **Cross-Platform Support**: Automatic detection and configuration for Local/Colab execution
- **Robust Error Handling**: Vocabulary validation, BERT truncation, memory management
- **Comprehensive Pipeline**: Data cleaning → Training → Evaluation → Compression

## Technical Highlights

### Innovation & Problem Solving

1. **Two-Stage Compression Architecture**
    - Stage 1: Fine-tuned MedGemma-4B generates verbose, high-recall summaries
    - Stage 2: Llama-3-8B compresses with entity preservation (Chain-of-Density inspired)
    - Sequential loading on single GPU (8GB VRAM) through aggressive memory management

2. **CUDA Error Prevention**
    - Discovered and fixed tokenizer/model vocabulary mismatch issue
    - Implemented validation checks and automatic embedding resizing
    - Proper loading order: Base Model → LoRA Adapters → Tokenizer

3. **BERTScore Truncation Solution**
    - Fixed BERT's 512 token limit overflow with proper tokenizer-based truncation
    - Uses `tokenizer.encode()` with truncation instead of naive word-count splitting
    - Ensures accurate metric computation on long clinical texts

4. **Memory Management Architecture**
    - Custom `flush_memory()` and `unload_model()` utilities
    - Complete model unloading between pipeline stages
    - Enables running 4B + 8B models sequentially on consumer hardware

5. **Clinical Domain Adaptation**
    - NumPy 1.x pinning for SciSpacy medical NER compatibility
    - Bio_ClinicalBERT for domain-specific semantic similarity
    - Instruction prompts emphasizing medical entity completeness

## Setup

### Requirements

Python 3.12+ with the following dependencies (managed via `uv`):

```bash
# Install dependencies using uv (recommended)
uv sync

# Or install using pip
pip install -r requirements.txt
```

### Execution Order

Follow this sequence to reproduce the full pipeline:

1. **Data Preparation**: Run `data_cleaning.ipynb` to preprocess MIMIC-IV dataset
2. **Model Training**: Run `clinical_discharge_summarization_peft.ipynb` to fine-tune MedGemma
3. **Evaluation** (Optional): Run `medgemma_model_evaluation.ipynb` to evaluate the model
4. **Compression** (Optional): Run `two_stage_compression_pipeline.ipynb` for compressed summaries

**Note**: Steps 3 and 4 are independent and can be run in any order after step 2.

### Notebook Summary

| Notebook                                      | Purpose                     | Input                     | Output                                                       | Runtime    |
|-----------------------------------------------|-----------------------------|---------------------------|--------------------------------------------------------------|------------|
| `data_cleaning.ipynb`                         | Preprocess MIMIC-IV dataset | Raw Kaggle dataset        | `mimic_cleaned_text_only.csv`                                | ~5-10 min  |
| `clinical_discharge_summarization_peft.ipynb` | Fine-tune MedGemma-4B       | Cleaned dataset           | LoRA adapters in `./medgemma-discharge-summarization/final/` | ~2-4 hours |
| `medgemma_model_evaluation.ipynb`             | Evaluate fine-tuned model   | LoRA adapters + test data | Metrics in `./evaluation_results/`                           | ~10-20 min |
| `two_stage_compression_pipeline.ipynb`        | Two-stage compression       | LoRA adapters + test data | Compressed summaries in `./compression_results/`             | ~30-60 min |

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

**Outputs**:

- Fine-tuned LoRA adapters saved to `./medgemma-discharge-summarization/final/`
- Includes adapter weights, config, and tokenizer files

#### 3. Model Evaluation (`medgemma_model_evaluation.ipynb`)

Evaluate the fine-tuned model independently:

- Loads the fine-tuned MedGemma model with LoRA adapters
- Generates predictions on test set
- Computes Clinical BERTScore metrics
- Performs qualitative analysis of generated summaries
- **Important**: Handles BERT's 512 token limit with proper truncation

**Outputs**:

- Evaluation results saved to `./evaluation_results/`
- CSV with predictions and scores
- JSON with summary statistics

#### 4. Two-Stage Compression Pipeline (`two_stage_compression_pipeline.ipynb`)

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
├── clinical_discharge_summarization_peft.ipynb   # MedGemma fine-tuning with QLoRA
├── medgemma_model_evaluation.ipynb               # Model evaluation with Clinical BERTScore
├── two_stage_compression_pipeline.ipynb          # Two-stage compression pipeline
├── pyproject.toml                                # Project dependencies (uv)
├── requirements.txt                              # Alternative pip requirements
├── uv.lock                                       # Locked dependencies
├── COMPRESSION_PIPELINE_OVERVIEW.md              # Compression pipeline documentation
└── README.md                                     # This file
```

**Generated Outputs**:

```
├── medgemma-discharge-summarization/
│   └── final/                                    # Fine-tuned LoRA adapters
├── evaluation_results/                           # Evaluation metrics and predictions
├── compression_results/                          # Compression pipeline outputs
├── mimic_cleaned_full.csv                        # Fully preprocessed dataset
└── mimic_cleaned_text_only.csv                   # Final training data
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

### 3. Model Evaluation (`medgemma_model_evaluation.ipynb`)

This notebook provides standalone evaluation of the fine-tuned MedGemma model:

**Purpose**:

- Independent evaluation separate from training pipeline
- Comprehensive metric computation on test set
- Qualitative analysis of generated summaries

**Model Loading**:

1. **Fine-tuned Model Setup**:
    - Loads base MedGemma-4B with 4-bit quantization
    - Applies saved LoRA adapters from training
    - Validates tokenizer-model vocabulary alignment
    - Includes CUDA error prevention with vocabulary size checks

2. **Memory Optimization**:
    - Uses QLoRA (4-bit) for efficient inference
    - CUDA synchronous mode for better error messages
    - Automatic embedding resizing if needed

**Evaluation Process**:

1. **Test Set Preparation**:
    - Loads preprocessed MIMIC dataset
    - 5% test split (consistent with training)
    - Configurable sample size for quick testing

2. **Prediction Generation**:
    - Generates summaries for all test samples
    - Uses same generation parameters as training
    - Tracks character counts and progress

3. **Clinical BERTScore Computation**:
    - Uses BERT's own tokenizer for accurate truncation (not word-based)
    - Truncates to 500 tokens to ensure no overflow
    - Compares predictions vs references using Bio_ClinicalBERT
    - Reports Precision, Recall, and F1 scores

**Text Truncation Solution**:

```python
# Uses BERT tokenizer to ensure texts fit within 512 token limit
tokens = tokenizer.encode(text, max_length=500, truncation=True)
truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
```

**Metrics Reported**:

- **Average Precision**: Clinical relevance of generated content
- **Average Recall**: Coverage of reference summary content (primary metric)
- **Average F1**: Harmonic mean of precision and recall
- **Per-sample scores**: Individual results for each test case

**Qualitative Analysis**:

- Side-by-side comparison of input, reference, and generated summaries
- First 3 examples shown with detailed scores
- Evaluation checklist for medical entity completeness

**Output Files**:

- `evaluation_results.csv`: Full results with predictions and scores
- `summary_statistics.json`: Aggregate metrics and configuration

**Configuration Options**:

- `NUM_TEST_SAMPLES`: Number of samples to evaluate (50 default, -1 for all)
- `MAX_NEW_TOKENS`: 512 for detailed summaries
- `TEMPERATURE`: 0.7 for balanced generation

**Required Libraries**:

- torch, transformers, peft, bitsandbytes, bert_score, numpy, pandas

### 4. Two-Stage Compression Pipeline (`two_stage_compression_pipeline.ipynb`)

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

- **Base Model**: google/medgemma-4b-it (4.3B parameters)
- **LoRA Configuration**: Rank 32, Alpha 64, Dropout 0.05
- **Trainable Parameters**: ~0.5% of total (highly parameter-efficient)
- **Quantization**: 4-bit NF4 with double quantization
- **Training**: 1 epoch, effective batch size 8 (gradient accumulation)
- **Max Sequence Length**: 2048 tokens (input), 512 tokens (generation)
- **Learning Rate**: 2e-4 with cosine schedule and 100 warmup steps

## Evaluation Metrics

The project uses clinical-specific evaluation metrics designed for medical text:

### Primary Metric: Clinical BERTScore

- Uses Bio_ClinicalBERT embeddings trained on MIMIC-III
- Measures semantic similarity beyond word overlap
- Captures medical terminology and clinical context
- Reports Precision, Recall, and F1 scores
- **Superior to BLEU/ROUGE** for medical text evaluation

### Secondary Metrics (Compression Pipeline)

- **Compression Ratio**: Measures summary length reduction
- **Entity Retention (NER)**: Medical entity preservation via SciSpacy
- **Clinical BERTScore**: Semantic similarity of compressed summaries

### Evaluation Philosophy

- **High Recall Focus**: Prioritizes capturing all medical entities
- **Target Recall**: ≥0.90 for production-ready summaries
- **Entity Completeness**: Verifies all diagnoses, medications, vitals, labs, procedures, and follow-up instructions

## Usage

After training, use the inference function to generate summaries for new clinical notes:

```python
summary = generate_discharge_summary(clinical_notes)
print(summary)
```

## Important Notes & Troubleshooting

### Memory Management

- **Single GPU Execution**: The two-stage compression pipeline is designed to run two 4B/8B models sequentially on a
  single GPU
- **VRAM Requirements**: Minimum 8GB VRAM recommended (tested on RTX 5060 with 8.55GB)
- **OOM Prevention**: Models are fully unloaded between stages using aggressive memory flushing

### Common Issues

**1. CUDA Error: Device-side assert triggered**

- **Cause**: Tokenizer/model vocabulary size mismatch
- **Solution**: The evaluation notebook includes automatic vocabulary validation and resizing
- **Prevention**: Always load LoRA adapters BEFORE tokenizer

**2. BERTScore RuntimeError: Tensor size mismatch**

- **Cause**: Clinical notes exceed BERT's 512 token limit
- **Solution**: Use BERT tokenizer for truncation (implemented in evaluation notebook)
- **Code**: `tokenizer.encode(text, max_length=500, truncation=True)`

**3. NumPy dtype size changed error**

- **Cause**: NumPy 2.x binary incompatibility with spacy/scispacy
- **Solution**: Use NumPy 1.x (specified in dependencies)

### HuggingFace Authentication

Some models require authentication:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Required for:

- Meta-Llama-3-8B-Instruct (compression pipeline)
- google/medgemma-4b-it (if access-restricted)

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

## Acknowledgments

### Dataset

- **MIMIC-IV BHC**: Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (
  version 2.2). PhysioNet.
- **Kaggle Dataset**: [BHC MIMIC-IV Summary](https://www.kaggle.com/datasets/aminexdr/bhc-mimic-iv-summary) by aminexdr

### Models

- **MedGemma**: Google's medical-domain adapted Gemma model
- **Bio_ClinicalBERT**: Alsentzer et al., "Publicly Available Clinical BERT Embeddings"
- **Meta-Llama-3**: Meta's open-source large language model
- **SciSpacy**: Medical NER models from Allen Institute for AI

### Frameworks & Libraries

- Hugging Face Transformers, PEFT, TRL, bitsandbytes
- PyTorch, Accelerate
- BERTScore, SciSpacy, NLTK

## Collaborators

- [Bryan Bergo](https://github.com/blbergo)
- [Mike Leon](https://github.com/mikeleon001)
- [Nicolas Escobedo](https://github.com/Nico25041)
- [Trung Vong](https://github.com/TrungVN9)
- [Talha Ahmed](https://github.com/talhaahmed884)

## Citation

If you use this project in your research or coursework, please cite:

```bibtex
@software{clinical_discharge_summarization_2025,
  title = {Clinical Discharge Summarization using MedGemma 4B with QLoRA},
  author = {Bergo, Bryan and Leon, Mike and Escobedo, Nicolas and Vong, Trung and Ahmed, Talha},
  year = {2025},
  note = {Gen AI Semester Project},
  url = {https://github.com/talhaahmed884/Gen-AI-Semester-Project-Patient-Hospital-Discharge-Summaries}
}
```

## License

This is an academic project for educational purposes. The code is released under MIT License. However, note that:

- **MIMIC-IV dataset** requires credentialed access and PhysioNet training
- **Model weights** are subject to their respective licenses (Gemma Terms of Use, Llama 3 Community License)
- **Clinical use**: This is a research prototype and NOT approved for clinical deployment