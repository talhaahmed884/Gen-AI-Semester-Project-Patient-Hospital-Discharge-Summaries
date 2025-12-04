# Two-Stage Clinical Summarization Compression Pipeline

## Executive Summary

This notebook implements Stage 2 of your clinical summarization course project: **compressing verbose clinical summaries
while preserving critical medical entities**.

## Pipeline Architecture

```
Clinical Notes
    ↓
[Stage 1: MedGemma-4B + LoRA]
    ↓
Verbose Summary (High Recall)
    ↓
[MEMORY UNLOAD - CRITICAL]
    ↓
[Stage 2: Llama-3-8B]
    ↓
Compressed Summary (Entity-Preserving)
    ↓
[Evaluation: 3 Metrics]
```

## Critical Design Decisions

### 1. Memory Management Strategy

**Problem**: Running two large models (4B + 8B parameters) on a single GPU (Google Colab T4: 15GB VRAM)

**Solution**: Sequential loading with aggressive memory flushing

```python
# After Stage 1 generation
unload_model(medgemma_model, medgemma_tokenizer)
flush_memory()  # gc.collect() + torch.cuda.empty_cache()

# Only then load Stage 2
llama_model = AutoModelForCausalLM.from_pretrained(...)
```

**Why This Matters**: Without complete unloading, you'll encounter OOM errors. The notebook includes explicit
verification of memory clearing between stages.

### 2. Chain-of-Density Inspired Compression

**Prompt Engineering**:

```
"Rewrite the following summary to be 50% shorter. 
You MUST retain ALL entities: medications (with dosages), 
vital signs (with numbers), lab results (with values), 
diagnoses, procedures, and dates. 
If you cannot shorten it without losing a critical fact, do not shorten it."
```

**Key Parameters**:

- Temperature: 0.3 (deterministic for reproducibility)
- Max tokens: 256 (forces conciseness)
- Target compression: ~50%

### 3. Environment Agnostic Execution

**Auto-Detection Logic**:

```python
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
# Mount Google Drive
# Use Drive paths
# Require HF authentication
else:
# Use local project paths
# Assume dependencies installed
```

**Colab-Specific Handling**:

- Automatic Drive mounting
- Configurable I/O paths (Section 2)
- HuggingFace authentication for Llama-3

**Local Execution**:

- Uses relative paths
- Saves to `./compression_results/`
- No Drive dependencies

## Evaluation Framework

### Metric 1: Compression Ratio

**Formula**: `len(stage2_summary) / len(stage1_summary)`

**Interpretation**:

- 0.50 = 50% compression (target)
- <0.50 = stronger compression
- > 0.50 = weaker compression

**Why It Matters**: Quantifies compression strength vs quality trade-off

### Metric 2: Entity Retention (NER)

**Method**:

1. Extract medical entities using SciSpacy (`en_core_sci_sm`)
2. Compare entities between Stage 1 and Stage 2
3. Calculate recall: `entities_retained / entities_original`

**Interpretation**:

- > 0.85 = Acceptable for clinical use
- <0.70 = Critical entities lost, unacceptable

**Why It Matters**: Losing medication dosages or lab values is clinically dangerous

### Metric 3: Clinical BERTScore

**Method**:

- Compare compressed summaries against **original clinical notes**
- Use Bio_ClinicalBERT embeddings (trained on MIMIC-III)
- Report Precision, Recall, F1

**Interpretation**:

- High F1 = Semantic similarity preserved despite compression
- Low F1 = Compression introduced semantic drift

**Why It Matters**: Traditional metrics (BLEU/ROUGE) fail for medical text due to synonyms and paraphrasing

## Output Files

After execution, you'll have:

1. **`compression_pipeline_results.csv`**
    - Columns: Source_Text, Stage1_Summary_Verbose, Stage2_Summary_Compressed, Compression_Ratio, Entity_Recall,
      BERTScore_Precision, BERTScore_Recall, BERTScore_F1
    - Use for detailed per-sample analysis

2. **`compression_pipeline_results.json`**
    - Same data in JSON format for programmatic analysis

3. **`summary_statistics.json`**
    - Aggregate metrics (mean, std, min, max)
    - Configuration snapshot (models used, parameters)

## Key Research Questions

The notebook includes analysis cells to investigate:

1. **Compression vs Quality Trade-off**
    - Does higher compression → lower entity recall?
    - Correlation analysis included

2. **BERTScore Validity**
    - Does Clinical BERTScore align with entity retention?
    - Or can summaries score high while missing critical entities?

3. **Failure Mode Analysis**
    - Which samples fail to compress?
    - Which lose the most entities?
    - Are there patterns in failures?

## Usage Instructions

### For Google Colab:

1. **Section 2**: Update these paths to match your Drive structure:
   ```python
   INPUT_DATA_PATH = "/content/drive/MyDrive/your-folder/mimic_cleaned_text_only.csv"
   MEDGEMMA_ADAPTER_PATH = "/content/drive/MyDrive/your-folder/medgemma-adapters/final"
   OUTPUT_DIR = "/content/drive/MyDrive/your-folder/compression_results"
   ```

2. **Section 7**: Authenticate with HuggingFace when prompted (required for Llama-3)

3. Run all cells sequentially

### For Local Execution:

1. Ensure all dependencies are installed:
   ```bash
   uv sync
   ```

2. Ensure you have:
    - `mimic_cleaned_text_only.csv` in project root
    - Fine-tuned MedGemma adapters in `./medgemma-discharge-summarization/final/`

3. Run all cells sequentially

## Memory Requirements

**Minimum**: 15GB GPU VRAM (Google Colab T4)

**Tested On**:

- Colab T4 (15GB): ✅ Works with 4-bit quantization
- Colab V100 (16GB): ✅ Works
- Local RTX 3090 (24GB): ✅ Works with headroom

**Will NOT Work On**:

- CPU-only machines (requires CUDA)
- GPUs <12GB VRAM

## Expected Runtime

For **10 samples** (default `NUM_SAMPLES = 10`):

- Stage 1 (MedGemma generation): ~2-3 minutes
- Model unloading: ~30 seconds
- Stage 2 (Llama compression): ~2-3 minutes
- Evaluation (NER + BERTScore): ~1-2 minutes
- **Total**: ~8-10 minutes

For **full dataset** (set `NUM_SAMPLES = -1`):

- Scales linearly with dataset size
- ~100 samples: ~1.5 hours

## Troubleshooting

### "CUDA Out of Memory"

**Cause**: Model not fully unloaded between stages

**Fix**:

1. Verify `unload_model()` was executed
2. Check `flush_memory()` output shows low allocation
3. Restart runtime if issue persists

### "File not found" (Colab)

**Cause**: Incorrect Drive paths in Section 2

**Fix**:

1. Navigate to file in Colab file browser (left sidebar)
2. Right-click → Copy path
3. Paste into `INPUT_DATA_PATH` variable

### "Module 'scispacy' not found"

**Cause**: SciSpacy model not installed

**Fix** (Colab):

```bash
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

## License

Academic project for educational purposes. Models used:

- MedGemma-4B: Google (Apache 2.0)
- Llama-3-8B: Meta (Llama 3 Community License)
- Bio_ClinicalBERT: MIT
