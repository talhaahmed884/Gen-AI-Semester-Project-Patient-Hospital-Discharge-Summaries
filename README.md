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

### Running the Notebook

The main workflow is in `clinical_discharge_summarization_peft.ipynb`:

- **Local execution**: Run cells starting from Section 3A (Local)
- **Google Colab**: Run Section 2A to mount Drive, then Section 3A (Colab)

## Project Structure

```
.
├── clinical_discharge_summarization_peft.ipynb  # Main training notebook
├── pyproject.toml                                # Project dependencies
└── README.md                                     # This file
```

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

The project uses the preprocessed MIMIC-IV BHC clinical discharge summaries dataset with columns:

- `final_input`: Clinical notes
- `final_target`: Reference discharge summaries

## Collaborators

- [Bryan Bergo](https://github.com/blbergo)
- [Mike Leon](https://github.com/mikeleon001)
- [Trung Nguyen](https://github.com/TrungVN9)
- [Talha Ahmed](https://github.com/talhaahmed884)

## License

This is an academic project for educational purposes.