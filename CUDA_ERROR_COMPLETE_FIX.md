# Complete CUDA Error Fix - Step by Step

## Problem

`AcceleratorError: CUDA error: device-side assert triggered` at `torch.multinomial`

This means **token IDs are out of range** for the model's vocabulary.

---

## Step 1: Add Diagnostic Cell (BEFORE Cell 15)

**Add this NEW cell right BEFORE the model loading cell**:

```python
# Enable synchronous CUDA for better error messages
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("✓ CUDA synchronous mode enabled")
```

This will give you more detailed error messages.

---

## Step 2: Replace Cell 15 Completely

**DELETE everything in Cell 15 and REPLACE with this**:

```python
print("=" * 80)
print("STAGE 1: LOADING MEDGEMMA-4B FOR VERBOSE SUMMARY GENERATION")
print("=" * 80)

# Step 1: Load base model FIRST (before tokenizer)
print(f"\nLoading base model: {MEDGEMMA_BASE_MODEL}")
print("  Quantization: 4-bit NF4")
print("  This may take 2-3 minutes...")

medgemma_model = AutoModelForCausalLM.from_pretrained(
    MEDGEMMA_BASE_MODEL,
    quantization_config=BNB_CONFIG,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
print("✓ Base model loaded")

# Step 2: Load LoRA adapters NEXT (before tokenizer)
if USE_MEDGEMMA_ADAPTER and os.path.exists(MEDGEMMA_ADAPTER_PATH):
    print(f"\nLoading LoRA adapters from: {MEDGEMMA_ADAPTER_PATH}")
    from peft import PeftModel

    medgemma_model = PeftModel.from_pretrained(medgemma_model, MEDGEMMA_ADAPTER_PATH)
    print("✓ LoRA adapters loaded")

    # CRITICAL: After loading LoRA, the model config may have updated vocab size
    # We need to use THIS vocab size for the tokenizer

elif USE_MEDGEMMA_ADAPTER:
    print(f"\n⚠️  WARNING: Adapter path not found: {MEDGEMMA_ADAPTER_PATH}")
    print("   Continuing with base model only")

# Step 3: Get the ACTUAL vocab size from the loaded model
embedding_layer = medgemma_model.get_input_embeddings()
actual_vocab_size = embedding_layer.weight.shape[0]
print(f"\n  Model embedding vocab size: {actual_vocab_size}")

# Step 4: Load tokenizer AFTER model is fully loaded
# Try adapter path first, then base model
if USE_MEDGEMMA_ADAPTER and os.path.exists(MEDGEMMA_ADAPTER_PATH):
    print(f"\nAttempting to load tokenizer from adapter path: {MEDGEMMA_ADAPTER_PATH}")
    try:
        medgemma_tokenizer = AutoTokenizer.from_pretrained(
            MEDGEMMA_ADAPTER_PATH,
            trust_remote_code=True,
            padding_side="right",
            add_eos_token=True
        )
        print("✓ Tokenizer loaded from adapter path")
    except Exception as e:
        print(f"⚠️  Adapter tokenizer failed: {e}")
        print("   Loading base model tokenizer instead")
        medgemma_tokenizer = AutoTokenizer.from_pretrained(
            MEDGEMMA_BASE_MODEL,
            trust_remote_code=True,
            padding_side="right",
            add_eos_token=True
        )
else:
    print(f"\nLoading tokenizer from base model: {MEDGEMMA_BASE_MODEL}")
    medgemma_tokenizer = AutoTokenizer.from_pretrained(
        MEDGEMMA_BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
        add_eos_token=True
    )

medgemma_tokenizer.pad_token = medgemma_tokenizer.eos_token

print(f"\n  Tokenizer vocab size: {len(medgemma_tokenizer)}")
print(f"  PAD token ID: {medgemma_tokenizer.pad_token_id}")
print(f"  EOS token ID: {medgemma_tokenizer.eos_token_id}")

# Step 5: CRITICAL VALIDATION - Check if sizes match
if len(medgemma_tokenizer) != actual_vocab_size:
    print(f"\n⚠️  MISMATCH DETECTED!")
    print(f"   Tokenizer vocab: {len(medgemma_tokenizer)}")
    print(f"   Model vocab: {actual_vocab_size}")

    if len(medgemma_tokenizer) > actual_vocab_size:
        print(f"\n   ERROR: Tokenizer is LARGER than model!")
        print(f"   This WILL cause CUDA errors!")
        print(f"\n   SOLUTION: Resizing model embeddings to {len(medgemma_tokenizer)}...")
        medgemma_model.resize_token_embeddings(len(medgemma_tokenizer))
        print(f"   ✓ Model embeddings resized")

        # Update actual vocab size
        actual_vocab_size = medgemma_model.get_input_embeddings().weight.shape[0]
        print(f"   ✓ New model vocab size: {actual_vocab_size}")
    else:
        print(f"\n   WARNING: Model is larger than tokenizer")
        print(f"   This is unusual but may work if all token IDs < {actual_vocab_size}")

# Step 6: VALIDATION TEST
print(f"\n{'=' * 80}")
print("CRITICAL VALIDATION TEST")
print(f"{'=' * 80}")

test_text = "Patient presented with chest pain."
test_tokens = medgemma_tokenizer(test_text, return_tensors="pt")
max_id = test_tokens['input_ids'].max().item()
min_id = test_tokens['input_ids'].min().item()

print(f"Test text: '{test_text}'")
print(f"Token IDs: {test_tokens['input_ids'][0].tolist()}")
print(f"Max token ID: {max_id}")
print(f"Min token ID: {min_id}")
print(f"Valid range: [0, {actual_vocab_size - 1}]")

if max_id >= actual_vocab_size:
    print(f"\n❌ CRITICAL ERROR: Token ID {max_id} >= vocab size {actual_vocab_size}")
    print(f"   This WILL cause the CUDA error you're seeing!")
    print(f"\n   IMMEDIATE FIX REQUIRED:")
    print(f"   medgemma_model.resize_token_embeddings({len(medgemma_tokenizer)})")
    raise ValueError(f"Token ID out of range: {max_id} >= {actual_vocab_size}")
elif min_id < 0:
    print(f"\n❌ CRITICAL ERROR: Negative token ID {min_id}")
    raise ValueError(f"Invalid token ID: {min_id}")
else:
    print(f"\n✅ VALIDATION PASSED!")
    print(f"   All token IDs are within valid range")
    print(f"   Safe to proceed with generation")

medgemma_model.eval()
print(f"\n✓ MedGemma ready for inference")
flush_memory()
```

---

## Step 3: Add Another Diagnostic Cell (AFTER Cell 15)

**Add this NEW cell RIGHT AFTER Cell 15**:

```python
# ============================================================================
# FINAL VALIDATION BEFORE GENERATION
# ============================================================================

print("=" * 80)
print("FINAL PRE-GENERATION CHECK")
print("=" * 80)

# Get actual embedding size
embedding_size = medgemma_model.get_input_embeddings().weight.shape[0]

print(f"\nTokenizer vocab: {len(medgemma_tokenizer)}")
print(f"Model embedding size: {embedding_size}")
print(f"Match: {'✅ YES' if len(medgemma_tokenizer) == embedding_size else '❌ NO'}")

# Test with actual prompt format from your generation function
test_prompt = """<start_of_turn>user
Summarize the following clinical discharge notes.

Clinical Notes:
Patient with hypertension.<end_of_turn>
<start_of_turn>model
"""

print(f"\nTesting with actual prompt format...")
test_inputs = medgemma_tokenizer(test_prompt, return_tensors="pt")
max_token = test_inputs['input_ids'].max().item()

print(f"  Prompt token count: {test_inputs['input_ids'].shape[1]}")
print(f"  Max token ID in prompt: {max_token}")
print(f"  Valid range: [0, {embedding_size - 1}]")

if max_token >= embedding_size:
    print(f"\n❌ STOP! Token ID {max_token} is out of range!")
    print(f"   DO NOT PROCEED - will cause CUDA error")
    print(f"\n   Run this fix:")
    print(f"   medgemma_model.resize_token_embeddings({len(medgemma_tokenizer)})")
else:
    print(f"\n✅ All checks passed - safe to generate summaries")

print("=" * 80)
```

---

## Step 4: If Validation Still Fails

If you STILL see errors in the validation, add this EMERGENCY FIX cell:

```python
# EMERGENCY FIX - Run this if validation fails
print("Running emergency embedding resize...")

# Get current sizes
tokenizer_size = len(medgemma_tokenizer)
embedding_size = medgemma_model.get_input_embeddings().weight.shape[0]

print(f"Tokenizer: {tokenizer_size}")
print(f"Model: {embedding_size}")

if tokenizer_size != embedding_size:
    print(f"\nResizing model to match tokenizer ({tokenizer_size})...")
    medgemma_model.resize_token_embeddings(tokenizer_size)
    print("✓ Resized")

    # Verify
    new_size = medgemma_model.get_input_embeddings().weight.shape[0]
    print(f"New model size: {new_size}")

    if new_size == tokenizer_size:
        print("✅ SUCCESS - sizes now match!")
    else:
        print("❌ FAILED - still mismatched")
else:
    print("✓ Already matched - no resize needed")
```

---

## What's Different in This Fix?

### Key Changes:

1. **Load model BEFORE tokenizer** (reverse order)
2. **Load LoRA adapters BEFORE tokenizer**
3. **Get vocab size from embedding layer** (most reliable)
4. **Multiple validation checks** with clear error messages
5. **Emergency stop** if validation fails

### Why This Order Matters:

- Loading LoRA adapters can change the model's vocab size
- We need to know the FINAL vocab size before loading tokenizer
- Then we ensure tokenizer matches this size

---

## Expected Output

When you run the updated Cell 15, you should see:

```
================================================================================
STAGE 1: LOADING MEDGEMMA-4B FOR VERBOSE SUMMARY GENERATION
================================================================================

Loading base model: google/medgemma-4b-it
  Quantization: 4-bit NF4
  This may take 2-3 minutes...
✓ Base model loaded

Loading LoRA adapters from: ./medgemma-discharge-summarization/final
✓ LoRA adapters loaded

  Model embedding vocab size: 262145

Attempting to load tokenizer from adapter path: ./medgemma-discharge-summarization/final
✓ Tokenizer loaded from adapter path

  Tokenizer vocab size: 262145
  PAD token ID: 1
  EOS token ID: 1

================================================================================
CRITICAL VALIDATION TEST
================================================================================
Test text: 'Patient presented with chest pain.'
Token IDs: [2, 28649, 5823, 675, 15489, 4380, 235265]
Max token ID: 28649
Min token ID: 2
Valid range: [0, 262144]

✅ VALIDATION PASSED!
   All token IDs are within valid range
   Safe to proceed with generation

✓ MedGemma ready for inference
GPU Memory: 0.05 GB allocated, 0.05 GB reserved
```

If you see `❌ CRITICAL ERROR`, **STOP** and run the emergency fix cell.

---

## Checklist

Before running generation:

- [ ] Cell 15 replaced with new code
- [ ] Validation cell added after Cell 15
- [ ] Validation shows `✅ VALIDATION PASSED`
- [ ] No error messages in validation output
- [ ] Tokenizer vocab == Model vocab

If ANY of these fail, **DO NOT proceed to generation** - the CUDA error will happen again.
