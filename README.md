# Technical Guide: Enhancing LLMs with RAG, CAG, and Fine-Tuning

This guide provides an educational overview and technical walkthrough of three key techniques used to enhance the capabilities of Large Language Models (LLMs): Retrieval-Augmented Generation (RAG), Cache-Augmented Generation (CAG), and Fine-Tuning. We'll explore the concepts behind each method, their implementation details, and guidance on when to choose each approach.

## Table of Contents

- [Introduction: Why Enhance LLMs?](#introduction-why-enhance-llms)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Cache-Augmented Generation (CAG)](#cache-augmented-generation-cag)
- [Fine-Tuning](#fine-tuning)

## Introduction: Why Enhance LLMs?

While large language models are incredibly powerful, they have limitations:

1.  **Knowledge Cutoffs:** Their internal knowledge is frozen at the time of training and doesn't include real-time information.
2.  **Hallucinations:** They can sometimes generate plausible but incorrect or fabricated information.
3.  **Generic Responses:** They might lack the specific domain knowledge or stylistic voice required for certain tasks.

RAG, CAG, and Fine-tuning are techniques designed to address these limitations by integrating external knowledge or adapting the model's behavior.

## Retrieval-Augmented Generation (RAG)

### Concept

RAG enhances LLM responses by dynamically retrieving relevant information snippets from an external knowledge base (like documents, databases, or websites) *at inference time*. This retrieved context is then provided to the LLM along with the user's query, allowing the model to generate more informed, accurate, and up-to-date answers grounded in the provided data.

### Pros & Cons

**Pros:**
*   **Access to Real-time Data:** Can incorporate the latest information without retraining.
*   **Reduced Hallucinations:** Grounds responses in factual, retrieved data.
*   **Transparency:** Easy to trace the source of information used in the response.
*   **Cost-Effective Updates:** Updating the knowledge base is cheaper than retraining.

**Cons:**
*   **Retrieval Latency:** The retrieval step adds time to the response generation.
*   **Retrieval Quality Dependency:** Performance heavily depends on the relevance and accuracy of the retrieved chunks.
*   **Context Window Limitations:** The amount of retrieved context is limited by the model's input capacity.

### When to Use RAG

*   When access to **up-to-date or rapidly changing information** is critical.
*   When **transparency and source attribution** are required.
*   For **knowledge-intensive tasks** where factual accuracy based on specific documents is paramount (e.g., customer support bots, Q&A over documentation).
*   When **computational resources for training are limited**.

### Implementation ([`rag.py`](https://github.com/mshojaei77/RAG_CAG_SFT/blob/main/rag.py))

This implementation uses LangChain and OpenAI to perform RAG on a local text file (`none-technical.md`).

**1. Setup and Initialization:**

- Import necessary libraries (`os`, `pathlib`, `hashlib`, `dotenv`, `langchain`, `openai`).
- Load environment variables (specifically `OPENAI_API_KEY`) using `dotenv`.
- Initialize `OpenAIEmbeddings` for creating text embeddings.
- Initialize `RecursiveCharacterTextSplitter` to break down the source document into manageable chunks.

```python
import os
from pathlib import Path
import hashlib
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY =  os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(separators=['.\n'],chunk_size=1500,chunk_overlap=50)

query = "What is Cache-Augmented Generation (CAG)?"
top_k = 2
file_path = 'none-technical.md'
```

**2. Indexing (Creating the Knowledge Base):**

- **Embedding Caching:**
    - Create an `embeddings` directory if it doesn't exist.
    - Calculate the MD5 hash of the input file (`none-technical.md`) to uniquely identify its content.
    - Define the path for the FAISS index file based on the hash (`embeddings/<hash>.faiss`).
- **Loading or Building the Index:**
    - Check if a FAISS index file for the current version of the document already exists.
    - **If exists:** Load the pre-computed index from the file using `FAISS.load_local`. This saves time on subsequent runs if the source document hasn't changed.
    - **If not exists:**
        - Read the text content of the source file (`none-technical.md`).
        - Split the text into documents (chunks) using the initialized `text_splitter`.
        - Create embeddings for each chunk using `OpenAIEmbeddings`.
        - Build a FAISS vector store from the documents and their embeddings using `FAISS.from_documents`.
        - Save the newly created FAISS index locally to the hash-based file path for future use.

```python
Path("embeddings").mkdir(exist_ok=True)
file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
embedding_file = Path("embeddings") / f"{file_hash}.faiss"

if embedding_file.exists():
    vector_db = FAISS.load_local(str(embedding_file), embeddings, allow_dangerous_deserialization=True)
else:
    text = Path(file_path).read_text(encoding='utf-8')
    documents = text_splitter.create_documents([text])
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(str(embedding_file))
```

**3. Retrieval:**

- Define the user query.
- Use the loaded/created `vector_db` (FAISS index) to perform a similarity search.
- `vector_db.similarity_search(query, k=top_k)` finds the `k` most relevant document chunks based on semantic similarity between the query embedding and the chunk embeddings.
- Extract the page content (the actual text) from the retrieved documents.

```python
result = vector_db.similarity_search(query, k=top_k)
contents = [doc.page_content for doc in result]
```

**4. Generation:**

- Initialize the OpenAI client.
- Construct a prompt that includes:
    - The original user query.
    - The retrieved `contents` as context.
- Use the `client.chat.completions.create` method to send the prompt to an OpenAI model (e.g., `gpt-4o-mini`).
- The LLM uses both its internal knowledge and the provided context (`contents`) to generate an answer.
- Print the response content.

```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
prompt = f"answer the query {query} base on following contents:\n {contents}"
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

response = r.choices[0].message.content
print(response)
```

### Key Components Used (RAG):

- **`langchain`:** Framework for orchestrating the RAG pipeline (text splitting, vector store interaction).
- **`openai`:** Library for interacting with OpenAI models (embeddings and generation).
- **`FAISS`:** Library for efficient similarity search in the vector store.
- **`dotenv`:** For managing API keys securely.
- **`hashlib`:** For creating a file hash to manage embedding persistence.

## Cache-Augmented Generation (CAG)

### Concept & Relation to KV Caching

CAG aims to improve performance and reduce latency for queries related to a **static knowledge base** by pre-processing this knowledge into the model's Key-Value (KV) cache *before* receiving user queries.

**What is the KV Cache?** During generation, LLMs use an attention mechanism. The KV cache stores intermediate computations (keys and values) from this mechanism for previously processed tokens. This avoids redundant calculations when generating subsequent tokens, speeding up inference significantly.

**How CAG Uses It:** CAG essentially pre-fills this KV cache with the representation of the entire static knowledge base. When a user query arrives, the model already has the knowledge "loaded" in its cache, eliminating the need for real-time retrieval (like RAG) or repeated processing of the same base knowledge. This precomputed cache is then reused for subsequent queries that fall within the scope of the preloaded knowledge.

Some platforms (like OpenAI, Anthropic) implement a similar optimization automatically, often called **Prompt Caching**, where they cache the KV state of frequently used prompt prefixes. CAG provides a way to explicitly control this preloading process, especially when using open-source models locally.

### Pros & Cons

**Pros:**
*   **Reduced Latency:** Eliminates the retrieval step, leading to faster responses compared to RAG for relevant queries.
*   **Consistency:** Responses are consistently based on the preloaded context.
*   **Simplified Architecture (Potentially):** Can remove the need for external vector databases if the knowledge fits entirely in the cache.

**Cons:**
*   **Static Knowledge:** Only works for knowledge bases that do not change frequently. Updates require recomputing the entire cache.
*   **Context Window/Memory Limits:** The entire knowledge base must fit within the model's context window and available hardware memory (even with quantization). This is a significant limitation.
*   **High Initial Cost:** Precomputing the cache can be computationally intensive initially.
*   **Potential Degradation:** Performance might degrade for extremely long contexts, even if they technically fit.

### When to Use CAG

*   When dealing with **stable, well-defined knowledge bases** that don't change often.
*   When **low latency** is a primary requirement.
*   When the **entire relevant knowledge can comfortably fit** within the model's context window and available memory.
*   For applications requiring **consistent responses** based on a fixed set of information (e.g., querying a specific manual or book).

### Implementation ([`cag.ipynb`](https://github.com/mshojaei77/RAG_CAG_SFT/blob/main/cag.ipynb) - Colab Notebook using Transformers)

This implementation uses the Hugging Face `transformers` library, `bitsandbytes` for quantization (to load larger models efficiently), and PyTorch.

**1. Setup and Model Loading:**

- Install necessary libraries, including `bitsandbytes` for 4-bit quantization.
- Import `torch` and relevant classes from `transformers` (`AutoTokenizer`, `BitsAndBytesConfig`, `AutoModelForCausalLM`, `DynamicCache`).
- **Quantization:** Define a `BitsAndBytesConfig` for 4-bit loading (NF4 type, double quantization) to reduce the model's memory footprint.
- **Model Initialization:** Load the tokenizer and the causal LM model (e.g., `huihui-ai/Llama-3.2-3B-Instruct-abliterated`) using the specified `quantization_config` and `device_map='auto'` for automatic device placement.
- **Login (Optional):** Use `notebook_login()` if accessing gated models from Hugging Face.

```python
!pip install -q -U bitsandbytes

import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM)
import bitsandbytes as bnb
from transformers.cache_utils import DynamicCache

# from huggingface_hub import notebook_login
# notebook_login()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

model_id  = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map='auto')
```

**2. Knowledge Loading:**

- Load the knowledge base content. In the Colab example, this is done via file upload, but it could also be read from a file path.
The content is stored in a string variable (e.g., `knowledge`).

```python
# Example: Loading knowledge from a string or file
# knowledge = "... your knowledge base content ..."
# Or using Colab file upload:
# from google.colab import files
# uploaded = files.upload()
# for fn in uploaded.keys():
#   knowledge = uploaded[fn].decode('utf-8')
```

**3. KV Cache Precomputation:**

- **`preprocess_knowledge` Function:**
    - Takes the model, tokenizer, and the knowledge prompt string.
    - Determines the embedding device from the model.
    - Encodes the knowledge prompt into `input_ids`.
    - Initializes an empty `DynamicCache` object to store the KV cache.
    - Performs a forward pass through the model (`model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, ...)`). The `use_cache=True` flag tells the model to compute and store the KV cache.
    - Returns the computed `past_key_values` (the KV cache) containing the representation of the knowledge prompt.
- **`prepare_kvcache` Function:**
    - Takes the raw knowledge documents and an optional instruction string.
    - Formats the knowledge into a structured prompt suitable for the model (e.g., using system and user tags).
    - Calls `preprocess_knowledge` to generate the KV cache for this formatted knowledge prompt.
    - Stores the cache (`kv`) and its sequence length (`kv_len`). This length is crucial for resetting the cache later.

```python
def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str) -> DynamicCache:
    embed_device = model.model.embed_tokens.weight.device
    input_ids    = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False)
    return outputs.past_key_values

def prepare_kvcache(documents, answer_instruction: str = None):
    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    # Format the knowledge with instructions/roles
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an ai assistant for giving short answers
    based on given documents.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """
    kv = preprocess_knowledge(model, tokenizer, knowledges)
    kv_len = kv.key_cache[0].shape[-2] # Get sequence length
    print("KV Cache Length:", kv_len)
    return kv, kv_len

# Precompute the cache
knowledge_cache, kv_len  = prepare_kvcache(documents=knowledge)
```

**4. Generation Using Precomputed Cache:**

- **`clean_up` Function:**
    - This function is essential for reusing the *precomputed* knowledge cache for multiple queries.
    - After a generation step, the cache might contain state related to the previous query+answer.
    - This function truncates the key and value tensors in the `past_key_values` object back to the original sequence length (`origin_len`) captured *after* the initial knowledge precomputation.
- **`generate` Function:**
    - Takes the model, tokenized user query (`input_ids`), the precomputed (and potentially cleaned) `past_key_values`, and `max_new_tokens`.
    - Iteratively generates tokens:
        - Passes the current `input_ids` (initially the query, then the last generated token) and the `past_key_values` to the model.
        - `use_cache=True` ensures the cache is used and updated.
        - Performs greedy decoding (selects the token with the highest probability using `argmax`).
        - Appends the new token to the output sequence.
        - Updates `past_key_values` with the output cache from the model step.
        - Stops when an EOS token is generated or `max_new_tokens` is reached.
    - Returns the generated token IDs (excluding the input query IDs).
- **Execution Flow:**
    - Define the user query string.
    - **Crucially, call `clean_up(knowledge_cache, kv_len)` before each new query** to reset the cache to the state containing only the precomputed knowledge.
    - Tokenize the user query.
    - Call the `generate` function, passing the model, tokenized query, and the cleaned `knowledge_cache`.
    - Decode the resulting token IDs into human-readable text.

```python
def clean_up(kv: DynamicCache, origin_len: int):
    # Truncate the cache back to its original precomputed length
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]

def generate(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int = 300
) -> torch.Tensor:
    embed_device = model.model.embed_tokens.weight.device
    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)
    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)
            past_key_values = outputs.past_key_values # Cache gets updated
            output_ids = torch.cat([output_ids, next_token], dim=1)

            # Check for EOS token (use model.config.eos_token_id)
            # Handle potential list/int differences in eos_token_id
            eos_ids = model.config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            if (next_token.item() in eos_ids) and (_ > 0):
                break
    return output_ids[:, origin_ids.shape[-1]:] # Return only generated tokens

# Example Query Execution
query = 'What is Cache-Augmented Generation (CAG)?'

# Reset cache before generating
clean_up(knowledge_cache, kv_len)

input_ids = tokenizer.encode(query, return_tensors="pt") # Don't move to device here
output = generate(model, input_ids, knowledge_cache)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Response of the model:\n {generated_text}")
```

### Key Components Used (CAG):

- **`transformers`:** Core library for accessing models (`AutoModelForCausalLM`), tokenizers (`AutoTokenizer`), and cache utilities (`DynamicCache`, `BitsAndBytesConfig`).
- **`torch`:** Tensor computation library used by `transformers`.
- **`bitsandbytes`:** Enables model quantization (e.g., 4-bit) to load large models on memory-constrained hardware.
- **KV Cache (`past_key_values`):** The core mechanism where the model stores intermediate attention computations. Precomputing this for the knowledge base is the essence of CAG.

### Limitations (CAG):

- The entire knowledge base must fit within the model's context window limit and available memory (even with quantization).
- The preloaded knowledge is static. Updates require recomputing the entire KV cache.
- Performance can degrade if the context becomes extremely long, even if it technically fits.

### Simpler Approach using APIs ([`simple.py`](https://github.com/mshojaei77/RAG_CAG_SFT/blob/main/simple.py))

An alternative, simpler way to achieve a conceptually similar result is shown in `simple.py`. This script uses the OpenAI API.

- It loads the entire knowledge base into the prompt's context, similar to how CAG prepares the context.
- However, it relies on the API provider (OpenAI in this case) to handle the underlying KV caching internally as an optimization.
- The user does not explicitly manage the cache; they simply send the full context with each query.
- This approach is easier to implement but offers less control over the caching mechanism and is still limited by the API's context window size and potential internal caching strategies.

## Fine-Tuning

### Concept

Fine-tuning adapts a pre-trained model to a specific task, domain, or style by continuing its training process on a custom, smaller dataset. Instead of just providing knowledge as context (like RAG/CAG), fine-tuning modifies the model's internal parameters (weights) to better align its behavior, knowledge, or output format with the target requirements.

### Pros & Cons

**Pros:**
*   **Improved Performance on Specific Tasks:** Can significantly outperform base models on specialized tasks.
*   **Style/Format Adaptation:** Effective at teaching the model a specific tone, voice, or output structure (e.g., JSON format).
*   **Domain Knowledge Integration:** Can embed specialized knowledge more deeply into the model.
*   **Potentially Faster Inference (Post-Training):** Once fine-tuned, inference can be faster than RAG as no retrieval is needed.

**Cons:**
*   **High Computational Cost & Time:** Requires significant resources and time for training.
*   **Requires Quality Datasets:** Performance heavily depends on the size and quality of the fine-tuning dataset.
*   **Risk of Catastrophic Forgetting:** The model might lose some of its general capabilities while specializing.
*   **Static Knowledge:** Doesn't inherently handle real-time data updates; requires retraining for new knowledge.
*   **Less Transparent:** Harder to trace *why* the model generated a specific output compared to RAG.

### When to Use Fine-Tuning

*   When you need to **adapt the model's style, tone, or output format**.
*   To **improve reliability on a specific, narrow task**.
*   To **imbue the model with specialized domain knowledge** that is relatively static.
*   When **high-quality, task-specific training data** is available.
*   When **computational resources for training are available**.

### Implementation Process

This section covers:
1.  Generating a Question/Answer dataset from a source document (`sft_data_prepare.py`).
2.  Fine-tuning a model (e.g., `gemma-3-4b-it`) using the generated dataset with `unsloth` and `trl` (based on the provided Colab notebook).

### 1. Dataset Preparation ([`sft_data_prepare.py`](https://github.com/mshojaei77/RAG_CAG_SFT/blob/main/sft_data_prepare.py))

This script automates the creation of a question-answer dataset in JSONL format, suitable for supervised fine-tuning.

**Process:**

- **Load Source Document:** Reads the content from a specified Markdown file (`INPUT_MD_FILE`).
- **Text Splitting:** Uses `LangChain`'s `RecursiveCharacterTextSplitter` to divide the document into smaller, manageable chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
- **Q&A Generation (per chunk):**
    - For each text chunk, it calls an OpenAI model (`OPENAI_MODEL`, e.g., `gpt-4o-mini`) via the `openai` library.
    - A specific system prompt instructs the model to generate a predefined number (`QA_PAIRS_PER_CHUNK`) of relevant Q&A pairs based *only* on the provided chunk text.
    - The prompt strongly requests the output to be *only* a JSON list of objects, each containing a `question` and `answer` key.
    - It uses the `response_format={"type": "json_object"}` feature for compatible models to encourage JSON output.
- **Validation & Error Handling:**
    - Includes robust error handling for API calls (retries using `MAX_RETRIES`).
    - Validates the received response to ensure it's valid JSON and conforms to the expected list-of-dictionaries structure (handling cases where the API might return a single object or wrap the list in a key).
    - Checks each Q&A pair within the list for the required `question` and `answer` keys.
    - Raises errors and stops execution if validation fails or maximum retries are exceeded.
- **Output:** Appends the validated Q&A pairs for each chunk to a JSONL file (`OUTPUT_JSONL_FILE`), where each line is a valid JSON object representing one Q&A pair.

**Key Components:**

- **`openai`:** For interacting with the LLM to generate Q&A pairs.
- **`langchain_text_splitters`:** For chunking the source document.
- **`dotenv`:** For managing the OpenAI API key.
- **`json`:** For handling JSON parsing and serialization.

**Configuration (Constants in the script):**

- `INPUT_MD_FILE`: Path to the source document.
- `OUTPUT_JSONL_FILE`: Path for the generated dataset.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parameters for text splitting.
- `OPENAI_MODEL`: The OpenAI model used for generation.
- `MAX_RETRIES`: Number of attempts for API calls.
- `QA_PAIRS_PER_CHUNK`: Target number of Q&A pairs per text chunk.

**To Run:**

1.  Ensure `none-technical.md` (or your desired input file) exists.
2.  Set your `OPENAI_API_KEY` in a `.env` file.
3.  Run the script: `python sft_data_prepare.py`
4.  The output `qa_pairs.jsonl` file will be created/updated.

### 2. Fine-Tuning with Unsloth ([`sft.ipynb`](https://github.com/mshojaei77/RAG_CAG_SFT/blob/main/sft.ipynb) - Colab Notebook)

This process uses the `unsloth` library for efficient fine-tuning, incorporating techniques like 4-bit quantization and LoRA (Low-Rank Adaptation).

**Process (Based on Colab):**

- **Setup & Installation:**
    - Installs necessary libraries: `unsloth`, `vllm`, `bitsandbytes`, `accelerate`, `peft`, `trl`, `datasets`, etc. Special handling for Colab dependencies is included.
    - Verifies CUDA availability.
- **Model Configuration & Loading:**
    - Uses `unsloth.FastModel` to load a base model (e.g., `unsloth/gemma-3-4b-it`).
    - Enables `load_in_4bit=True` for 4-bit quantization to reduce memory usage.
    - Applies PEFT (Parameter-Efficient Fine-Tuning) using `FastModel.get_peft_model`:
        - Configures LoRA parameters (`r`, `lora_alpha`, `lora_dropout`).
        - Specifies which parts of the model to fine-tune (language layers, attention, MLP modules).
- **Dataset Loading & Formatting:**
    - Uploads the generated `qa_pairs.jsonl` file (e.g., using Colab's `files.upload`).
    - Loads the dataset using `datasets.load_dataset`.
    - **Formatting:**
        - Defines a function (`format_to_chat`) to transform each Q&A pair into the required chat format: `{"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}`.
        - Maps this function to the dataset.
        - Uses `unsloth.chat_templates.standardize_data_formats` for consistency.
        - Applies the model's specific chat template using `tokenizer.apply_chat_template` to create a single `text` field for training.
- **Training:**
    - Initializes `trl.SFTTrainer` (Supervised Fine-tuning Trainer):
        - Passes the configured `model` and `tokenizer`.
        - Provides the formatted `train_dataset`.
        - Sets training arguments using `trl.SFTConfig`:
            - `dataset_text_field="text"`
            - Batch size (`per_device_train_batch_size`, `gradient_accumulation_steps`)
            - Learning rate, optimizer (`adamw_8bit`), scheduler, warmup steps.
            - `max_steps` (for shorter runs) or `
