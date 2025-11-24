# Qwen3-4B Fine-Tuned Dialogue Summarizer

This model is fine-tuned on the **SAMSum** dataset for abstractive dialogue summarization. It leverages the reasoning capabilities of the **Qwen3-4B** architecture to generate concise, factual, and coherent summaries of real-world conversations.

## Model Description

- **Model Type**: Qwen3-4B (Causal Language Model) with 4-bit Quantization
- **Task**: Abstractive Dialogue Summarization
- **Dataset**: [SAMSum Corpus](https://huggingface.co/datasets/knkarthick/samsum)
- **Training Objective**: Summarize informal messanger-like conversations into third-person narratives.
- **Key Features**:
    - **Thinking Mode**: Inherits Qwen3's ability to reason through complex dialogues before summarizing.
    - **Efficiency**: Optimized with Unsloth's 4-bit quantization for low-memory inference.
    - **Architecture**: Causal Decoder-only Transformer.

## Model Link on Hugging Face

You can find the model here: [Qwen3-4B Dialogue Summarizer on Hugging Face](https://huggingface.co/riturajpandey739/Qwen3-4B-Text-Summarizer-finetuning)

## Intended Use

This model is designed to summarize:
- Chat logs (WhatsApp, Slack, Discord).
- Meeting transcripts.
- Customer support dialogues.

It is specifically trained on English conversations but benefits from Qwen3's underlying multilingual capabilities.

## Model Details

- **Base Architecture**: Qwen/Qwen3-4B-Instruct (4 Billion Parameters)
- **Input**: Text (Conversation/Dialogue)
- **Output**: Text (Summary)
- **Context Length**: Supports up to **32,768 tokens** natively.
- **Training Data**: The model was trained using the **SAMSum** dataset, which contains ~16k chat dialogues annotated with summaries.
- **Fine-tuning Method**:
    - **Framework**: Unsloth + TRL + PEFT.
    - **Technique**: Rank-Stabilized LoRA (RSLoRA) with `r=64` and `alpha=32`.
    - **Target Modules**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

## Performance

The model utilizes Qwen3's **"Thinking Mode"** capabilities to better grasp speaker intent and implicit meaning compared to standard summarizers.
* *Note: Specific ROUGE scores depend on the evaluation set and generation parameters.*

## Limitations and Biases

- **Language**: The fine-tuning data (SAMSum) is primarily in English, so performance on other languages may be degraded despite the base model's multilingual support.
- **Format**: The model is optimized for dialogue format (`Speaker A: Text`). It may struggle with standard article or book summarization.
- **Bias**: As with any LLM, it may reflect biases present in the training data or the base pre-training corpus.

## How to Use

To use the model with the Hugging Face Transformers library, follow these steps. 

**Note:** Since Qwen3 is a Causal LM (Decoder-only), we use the `text-generation` pipeline, not `summarization`.

```python
from transformers import pipeline

# 1. Load the pipeline
# We use "text-generation" because Qwen is a Causal LM.
# model_kwargs ensures it loads in 4-bit to save memory.
summarizer = pipeline(
    "text-generation",
    model="riturajpandey739/Qwen3-4B-Text-Summarizer-finetuning",
    model_kwargs={"load_in_4bit": True, "device_map": "auto"}
)

# 2. Define the input with the Prompt Template used in training
conversation = """
Manager: The server is down. We are losing clients.
Engineer: I identified the bug. It's a memory leak in the new deployment.
Manager: How long to fix?
Engineer: I am rolling back the update now. Should be up in 10 mins.
"""

prompt = f"""Below is a conversation between people. Write a concise summary of the conversation.

### Dialogue:
{conversation}

### Summary:
"""

# 3. Generate Summary
result = summarizer(
    prompt,
    max_new_tokens=128,
    return_full_text=False, # Set to False to return only the summary
    temperature=0.1         # Low temperature for factual consistency
)

# 4. Output the result
print(result[0]['generated_text'])
# Example Output: "The Engineer informs the Manager that the server is down due to a memory leak. They are rolling back the update and expect it to be fixed in 10 minutes."
