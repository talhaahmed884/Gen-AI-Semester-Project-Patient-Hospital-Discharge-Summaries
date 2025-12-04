# Project Dependencies

This document provides a comprehensive overview of all dependencies used across the project notebooks.

## Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt

# Optional: Install SciSpacy medical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

## Dependency Matrix

| Package                          | Version         | Used In                                     | Purpose                                                     |
|----------------------------------|-----------------|---------------------------------------------|-------------------------------------------------------------|
| **Core Data Science**            |                 |                                             |                                                             |
| numpy                            | >=1.26.0,<2.0.0 | All notebooks                               | Numerical computing (pinned to 1.x for spacy compatibility) |
| pandas                           | >=2.2.3         | All notebooks                               | Data manipulation and analysis                              |
| **Data Acquisition**             |                 |                                             |                                                             |
| kagglehub                        | >=0.3.8         | data_cleaning.ipynb                         | Download MIMIC-IV dataset from Kaggle                       |
| **Natural Language Processing**  |                 |                                             |                                                             |
| nltk                             | >=3.9.1         | data_cleaning.ipynb                         | Tokenization, stopword removal, lemmatization               |
| spacy                            | >=3.7.0,<3.8.0  | two_stage_compression_pipeline.ipynb        | Medical NER for entity extraction                           |
| scispacy                         | >=0.5.4         | two_stage_compression_pipeline.ipynb        | Medical NER models (en_core_sci_sm)                         |
| **Data Visualization**           |                 |                                             |                                                             |
| matplotlib                       | >=3.9.0         | data_cleaning.ipynb                         | Plotting and visualization                                  |
| seaborn                          | >=0.13.2        | data_cleaning.ipynb                         | Statistical data visualization                              |
| **Deep Learning & Transformers** |                 |                                             |                                                             |
| torch                            | >=2.9.1         | Training, Evaluation, Compression           | PyTorch deep learning framework                             |
| transformers                     | >=4.57.3        | Training, Evaluation, Compression           | HuggingFace model library                                   |
| peft                             | >=0.18.0        | Training, Evaluation                        | Parameter-Efficient Fine-Tuning (LoRA)                      |
| bitsandbytes                     | >=0.48.2        | Training, Evaluation, Compression           | 4-bit quantization (QLoRA)                                  |
| trl                              | >=0.25.1        | clinical_discharge_summarization_peft.ipynb | Supervised Fine-Tuning Trainer                              |
| accelerate                       | >=1.12.0        | Training, Evaluation, Compression           | Distributed training and device mapping                     |
| **ML Datasets & Utilities**      |                 |                                             |                                                             |
| datasets                         | >=4.4.1         | Training, Evaluation                        | HuggingFace datasets library                                |
| huggingface-hub                  | >=0.26.0        | Training, Compression                       | Model download and authentication                           |
| **Evaluation Metrics**           |                 |                                             |                                                             |
| bert-score                       | >=0.3.13        | Evaluation, Compression                     | Clinical BERTScore evaluation                               |
| scipy                            | >=1.16.3        | Training, Evaluation                        | Scientific computing utilities                              |
| **Model Architecture Utilities** |                 |                                             |                                                             |
| einops                           | >=0.8.1         | Training                                    | Tensor operations for Gemma architecture                    |
| **Jupyter Notebook Support**     |                 |                                             |                                                             |
| ipykernel                        | >=7.1.0         | All notebooks                               | Jupyter kernel                                              |
| ipywidgets                       | >=8.1.8         | All notebooks                               | Interactive widgets                                         |

## Notebook-Specific Dependencies

### 1. data_cleaning.ipynb

```python
# Core
import pandas as pd
import numpy as np
import re
from collections import Counter

# Data Acquisition
import kagglehub
from kagglehub import KaggleDatasetAdapter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

**Additional Setup Required:**

```python
# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 2. clinical_discharge_summarization_peft.ipynb

```python
# Core
import warnings
import numpy as np
import pandas as pd
import torch

# Transformers & PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Datasets & Evaluation
from datasets import Dataset
from bert_score import BERTScorer

# Utilities
from huggingface_hub import notebook_login
```

### 3. medgemma_model_evaluation.ipynb

```python
# Core
import warnings
import numpy as np
import pandas as pd
import torch
import json
import os

# Transformers & PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# Datasets & Evaluation
from datasets import Dataset
from bert_score import BERTScorer
```

### 4. two_stage_compression_pipeline.ipynb

```python
# Core
import gc
import json
import warnings
import numpy as np
import pandas as pd
import torch
import sys
import os

# Transformers & PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# NLP & Evaluation
import spacy
from bert_score import BERTScorer

# Utilities
from huggingface_hub import notebook_login
```

**Additional Setup Required:**

```python
# Install SciSpacy medical NER model
!pip
install
https: // s3 - us - west - 2.
amazonaws.com / ai2 - s2 - scispacy / releases / v0
.5
.4 / en_core_sci_sm - 0.5
.4.tar.gz
```

## Platform-Specific Dependencies

### Google Colab Only

```python
from google.colab import drive  # Not needed for local execution
```

### System Utilities (for monitoring)

```python
import psutil  # Used in Colab for RAM monitoring
import subprocess  # Used for nvidia-smi checks
```

## Critical Version Constraints

### NumPy Version Pinning

```
numpy>=1.26.0,<2.0.0
```

**Reason**: NumPy 2.x has binary incompatibility with spacy/scispacy. Using NumPy 2.x will cause:

```
ValueError: numpy.dtype size changed, may indicate binary incompatability
```

### Spacy Version Constraint

```
spacy>=3.7.0,<3.8.0
```

**Reason**: Compatibility with scispacy v0.5.4 and medical NER models.

## Optional Dependencies

### Development Tools

```
jupyter>=1.0.0
jupyterlab>=4.0.0
```

Install with:

```bash
uv sync --extra dev
```

## Hardware Requirements

### CUDA/GPU

- **CUDA Toolkit**: 11.8 or higher (for PyTorch 2.9.1+cu130)
- **VRAM**: Minimum 8GB recommended
- **Tested on**: NVIDIA RTX 5060 (8.55GB VRAM)

### CPU/RAM

- **RAM**: 16GB minimum recommended
- **CPU**: Multi-core processor for data preprocessing

## Troubleshooting

### Issue: NumPy dtype error with spacy

**Error**: `ValueError: numpy.dtype size changed`
**Solution**: Ensure numpy is pinned to 1.x: `pip install "numpy>=1.26.0,<2.0.0"`

### Issue: bitsandbytes CUDA errors

**Error**: `CUDA extension not compiled with the right CUDA version`
**Solution**: Install bitsandbytes from source or use pre-built wheels for your CUDA version

### Issue: Out of Memory (OOM) errors

**Solution**:

- Reduce batch size in training
- Use gradient accumulation
- Ensure models are unloaded between pipeline stages (compression notebook)

### Issue: HuggingFace authentication required

**Solution**:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Required for: Meta-Llama-3-8B-Instruct, google/medgemma-4b-it (if access-restricted)

## Dependency Updates

To update dependencies:

```bash
# Using uv
uv lock --upgrade

# Using pip
pip install --upgrade -r requirements.txt
```

## Minimal Installation (Testing Only)

For quick testing without full pipeline:

```bash
pip install torch transformers pandas numpy jupyter
```

Note: This minimal installation only supports basic model inference, not training or evaluation.
