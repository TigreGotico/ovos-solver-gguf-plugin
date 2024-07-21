# GGUF Solver

## Overview

`GGUFSolver` is a question-answering module that utilizes GGUF models to provide responses to user queries. This solver
streams utterances for real-time interaction and is built on the `ovos_plugin_manager.templates.solvers.QuestionSolver`
framework.

## Features

- Supports loading GGUF models from local files or remote repositories.
- Streams partial responses for improved interactivity.
- Configurable persona and verbosity settings.
- Capable of providing spoken answers.

## Configuration

`GGUFSolver` requires a configuration dictionary. The configuration should at least specify the model to use. Here is an
example configuration:

```python
cfg = {
    "model": "TheBloke/notus-7B-v1-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
```

- `model`: The identifier for the model. It can be a local file path or a repository ID for a remote model.
- `n_gpu_layers`: how many layer to offload to GPU, `-1` to offload all. default `0`
- `remote_filename`: The specific filename to load from a remote repository.
- `chat_format`: (Optional) Chat formatting settings.
- `verbose`: (Optional) Set to `True` for detailed logging.
- `persona`: (Optional) Persona for the system messages. Default
  is `"You are a helpful assistant who gives short factual answers"`.
- `max_tokens`: (Optional) Maximum tokens for the response. Default is `512`.

**NOTE**: for GPU support llama.cpp needs to be compiled with

`CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir`

## Usage

### Initializing the Solver

```python
from ovos_gguf_solver import GGUFSolver
from ovos_utils.log import LOG

LOG.set_level("DEBUG")

cfg = {
    "model": "TheBloke/notus-7B-v1-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}

solver = GGUFSolver(cfg)
```

### Streaming Utterances

Use the `stream_utterances` method to stream responses. This is particularly useful for real-time applications such as
voice assistants.

```python
query = "tell me a joke about aliens"
for sentence in solver.stream_utterances(query):
    print(sentence)
```

### Getting a Full Answer

Use the `get_spoken_answer` method to get a complete response.

```python
query = "What is the capital of France?"
answer = solver.get_spoken_answer(query)
print(answer)
```

## Integrating with Persona Framework

To integrate `GGUFSolver` with the OVOS Persona Framework and pass solver configurations, follow these examples.

Each example demonstrates how to define a persona configuration file with specific settings for different models or configurations.

To use any of these configurations, run the OVOS Persona Server with the desired configuration file:

```bash
$ ovos-persona-server --persona gguf_persona_remote.json
```

Replace `gguf_persona_remote.json` with the filename of the configuration you wish to use.

### Example 1: Using a Remote GGUF Model

This example shows how to configure the `GGUFSolver` to use a remote GGUF model with a specific persona.

**`gguf_persona_remote.json`**:

```json
{
  "name": "Notus",
  "solvers": [
    "ovos-solver-gguf-plugin"
  ],
  "ovos-solver-gguf-plugin": {
    "model": "TheBloke/notus-7B-v1-GGUF",
    "remote_filename": "*Q4_K_M.gguf",
    "persona": "You are an advanced assistant providing detailed and accurate information.",
    "verbose": true
  }
}
```

In this configuration:
- `ovos-solver-gguf-plugin` is set to use a remote GGUF model `TheBloke/notus-7B-v1-GGUF` with the specified filename.
- The persona is configured to provide detailed and accurate information.
- `verbose` is set to `true` for detailed logging.

### Example 2: Using a Local GGUF Model

This example shows how to configure the `GGUFSolver` to use a local GGUF model.

**`gguf_persona_local.json`**:

```json
{
  "name": "LocalGGUFPersona",
  "solvers": [
    "ovos-solver-gguf-plugin"
  ],
  "ovos-solver-gguf-plugin": {
    "model": "/path/to/local/model/gguf_model.gguf",
    "persona": "You are a helpful assistant providing concise answers.",
    "max_tokens": 256
  }
}
```

In this configuration:
- `ovos-solver-gguf-plugin` is set to use a local GGUF model located at `/path/to/local/model/gguf_model.gguf`.
- The persona is configured to provide concise answers.
- `max_tokens` is set to `256` to limit the response length.


### Example Models

these models are not endorsed and this list was largely compiling by searching hugging face, only for illustrative
purposes

| Language   | Model Name                                          | URL                                                                                              | Description                                                                                                                                                                                                                                                                                                                                                                                             |
|------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| English    | CausalLM-14B-GGUF                                   | [Link](https://huggingface.co/TheBloke/CausalLM-14B-GGUF)                                        | A 14B parameter model compatible with Meta LLaMA 2, demonstrating top-tier performance among models with fewer than 70B parameters, optimized for both qualitative and quantitative evaluations, with strong consistency across versions.                                                                                                                                                               |
| English    | Phi-3-Mini-4K-Instruct-GGUF                         | [Link](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)                             | A lightweight 3.8B parameter model from the Phi-3 family, optimized for strong reasoning and long-context tasks with robust performance in instruction adherence and logical reasoning.                                                                                                                                                                                                                 |
| English    | Qwen2-0.5B-Instruct-GGUF                            | [Link](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF)                                     | A 0.5B parameter instruction-tuned model from the Qwen2 series, excelling in language understanding, generation, and multilingual tasks with competitive performance against state-of-the-art models.                                                                                                                                                                                                   |
| English    | GritLM_-_GritLM-7B-gguf                             | [Link](https://huggingface.co/RichardErkhov/GritLM_-_GritLM-7B-gguf)                             | A unified model for both text generation and embedding tasks, achieving state-of-the-art performance in both areas and enhancing Retrieval-Augmented Generation (RAG) efficiency by over 60%.                                                                                                                                                                                                           |
| English    | falcon-7b-instruct-GGUF                             | [Link](https://huggingface.co/QuantFactory/falcon-7b-instruct-GGUF)                              | A 7B parameter instruct model based on Falcon-7B, optimized for chat and instruction tasks with performance benefits from extensive training on 1,500B tokens, and optimized inference architecture.                                                                                                                                                                                                    |
| English    | Samantha-Qwen-2-7B-GGUF                             | [Link](https://huggingface.co/QuantFactory/Samantha-Qwen-2-7B-GGUF)                              | A quantized 7B parameter model fine-tuned with QLoRa and FSDP, tailored for conversational tasks and utilizing datasets like OpenHermes-2.5 and Opus_Samantha.                                                                                                                                                                                                                                          |
| English    | Mistral-7B-Instruct-v0.3-GGUF                       | [Link](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF)                       | An instruct fine-tuned model based on Mistral-7B-v0.3, featuring an extended vocabulary and support for function calling, aimed at demonstrating effective fine-tuning with room for improved moderation mechanisms.                                                                                                                                                                                    |
| English    | Lite-Mistral-150M-v2-Instruct-GGUF                  | [Link](https://huggingface.co/OuteAI/Lite-Mistral-150M-v2-Instruct-GGUF)                         | A compact 150M parameter model optimized for efficiency on various devices, demonstrating reasonable performance in simple queries but facing challenges with context preservation and accuracy in multi-turn conversations.                                                                                                                                                                            |
| English    | TowerInstruct-7B-v0.1-GGUF                          | [Link](https://huggingface.co/TheBloke/TowerInstruct-7B-v0.1-GGUF)                               | A 7B parameter model fine-tuned on the TowerBlocks dataset for translation tasks, including general, context-aware, and terminology-aware translation, as well as named-entity recognition and grammatical error correction.                                                                                                                                                                            |
| English    | Dr_Samantha-7B-GGUF                                 | [Link](https://huggingface.co/TheBloke/Dr_Samantha-7B-GGUF)                                      | A merged model incorporating medical and psychological knowledge, with extensive performance on medical knowledge tasks and a focus on whole-person care.                                                                                                                                                                                                                                               |
| English    | phi-2-orange-GGUF                                   | [Link](https://huggingface.co/rhysjones/phi-2-orange)                                            | A finetuned model based on Phi-2, optimized with a two-step finetuning approach for improved performance in various evaluation metrics. The model is designed for Python-related tasks and general question answering.                                                                                                                                                                                  |
| English    | phi-2-electrical-engineering-GGUF                   | [Link](https://huggingface.co/TheBloke/phi-2-electrical-engineering-GGUF)                        | The phi-2-electrical-engineering model excels in answering questions and generating code specifically for electrical engineering and Kicad software, boasting efficient deployment and a focus on technical accuracy within its 2.7 billion parameters.                                                                                                                                                 |
| English    | Unholy-v2-13B-GGUF                                  | [Link](https://huggingface.co/TheBloke/Unholy-v2-13B-GGUF)                                       | An uncensored 13B parameter model merged with various models for an uncensored experience, designed to bypass typical content moderation filters.                                                                                                                                                                                                                                                       |
| English    | CapybaraHermes-2.5-Mistral-7B-GGUF                  | [Link](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF)                       | A preference-tuned 7B model using distilabel, optimized for multi-turn performance with improved scores in benchmarks like MTBench and Nous, compared to the Mistral-7B-Instruct-v0.2.                                                                                                                                                                                                                  |
| English    | notus-7B-v1-GGUF                                    | [Link](https://huggingface.co/TheBloke/notus-7B-v1-GGUF)                                         | A 7B parameter model fine-tuned with Direct Preference Optimization (DPO), surpassing Zephyr-7B-beta and Claude 2 on AlpacaEval, designed for chat-like applications with improved preference-based performance.                                                                                                                                                                                        |
| English    | Luna AI Llama2 Uncensored GGML                      | [Link](https://huggingface.co/TheBloke/Luna-AI-Llama2-Uncensored-GGML)                           | A Llama2-based chat model fine-tuned on over 40,000 long-form chat discussions. Optimized with synthetic outputs, available in both 4-bit GPTQ for GPU and GGML for CPU inference. Prompt format follows Vicuna 1.1/OpenChat style.                                                                                                                                                                     |
| English    | Zephyr-7B-β-GGUF                                    | [Link](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF)                                      | A 7B parameter model fine-tuned with Direct Preference Optimization (DPO) to enhance performance, optimized for helpfulness but may generate problematic text due to removed in-built alignment.                                                                                                                                                                                                        |
| English    | TinyLlama-1.1B-1T-OpenOrca-GGUF                     | [Link](https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF)                          | A 1.1B parameter model fine-tuned on the OpenOrca GPT-4 subset, optimized for conversational tasks with a focus on efficiency and performance in the CHATML format.                                                                                                                                                                                                                                     |
| English    | LlongOrca-7B-16K-GGUF                               | [Link](https://huggingface.co/TheBloke/LlongOrca-7B-16K-GGUF)                                    | A fine-tuned 7B parameter model optimized for long contexts, achieving top performance in long-context benchmarks and notable improvements over the base model, with efficient training using OpenChat's MultiPack algorithm.                                                                                                                                                                           |
| English    | Meta-Llama-3-8B-Instruct-GGUF                       | [Link](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)                        | An 8B parameter instruction-tuned model from the Llama 3 series, optimized for dialogue and outperforming many open-source models on industry benchmarks, with a focus on helpfulness and safety through advanced fine-tuning techniques.                                                                                                                                                               |
| English    | Smol-7B-GGUF                                        | [Link](https://huggingface.co/TheBloke/smol-7B-GGUF)                                             | A fine-tuned 7B parameter model from the Smol series, known for its strong performance in diverse NLP tasks and efficient fine-tuning techniques.                                                                                                                                                                                                                                                       |
| English    | Smol-Llama-101M-Chat-v1-GGUF                        | [Link](https://huggingface.co/afrideva/Smol-Llama-101M-Chat-v1-GGUF)                             | A compact 101M parameter chat model optimized for diverse conversational tasks, showing balanced performance across multiple benchmarks with a focus on efficiency and low-resource scenarios.                                                                                                                                                                                                          |
| English    | Sonya-7B-GGUF                                       | [Link](https://huggingface.co/TheBloke/Sonya-7B-GGUF)                                            | A high-performing 7B model with excellent scores in MT-Bench, ideal for various tasks including assistant and roleplay, combining multiple sources to achieve superior performance.                                                                                                                                                                                                                     |
| English    | WizardLM-7B-uncensored-GGML                         | [Link](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML)                              | An uncensored 7B parameter model from the WizardLM series, designed without built-in alignment to allow for custom alignment via techniques like RLHF LoRA, with no guardrails and complete responsibility for content usage resting on the user.                                                                                                                                                       |
| English    | OpenChat 3.5                                        | [Link](https://huggingface.co/TheBloke/openchat_3.5-GGUF)                                        | A 7B parameter model that achieves comparable results with ChatGPT, excelling in MT-bench evaluations. It utilizes mixed-quality data and C-RLFT (a variant of offline reinforcement learning) for training. OpenChat 3.5 performs well across various benchmarks and has been optimized for high-throughput deployment. It is an open-source model with strong performance in chat-based applications. |
| Portuguese | PORTULAN_-_gervasio-7b-portuguese-ptpt-decoder-gguf | [Link](https://huggingface.co/RichardErkhov/PORTULAN_-_gervasio-7b-portuguese-ptpt-decoder-gguf) | Gervásio 7B PTPT is an open decoder for Portuguese, built on the LLaMA-2 7B model, fine-tuned with instruction data to excel in various Portuguese tasks, and designed to run on consumer-grade hardware with a focus on European Portuguese.                                                                                                                                                           |
| Portuguese | CabraLlama3-8b-GGUF                                 | [Link](https://huggingface.co/mradermacher/CabraLlama3-8b-GGUF)                                  | A refined version of Meta-Llama-3-8B-Instruct, optimized with the Cabra 30k dataset for understanding and responding in Portuguese, providing enhanced performance for Portuguese language tasks.                                                                                                                                                                                                       |
| Portuguese | bode-7b-alpaca-pt-br-gguf                           | [Link](https://huggingface.co/recogna-nlp/bode-7b-alpaca-pt-br-gguf)                             | Bode-7B is a fine-tuned LLaMA 2-based model designed for Portuguese, delivering satisfactory results in classification tasks and prompt-based applications.                                                                                                                                                                                                                                             |
| Portuguese | bode-13b-alpaca-pt-br-gguf                          | [Link](https://huggingface.co/recogna-nlp/bode-13b-alpaca-pt-br-gguf)                            | Bode-13B is a fine-tuned LLaMA 2-based model for Portuguese prompts, offering enhanced performance over its 7B counterpart, and designed for both research and commercial applications with a focus on Portuguese language tasks.                                                                                                                                                                       |
| Portuguese | sabia-7B-GGUF                                       | [Link](https://huggingface.co/TheBloke/sabia-7B-GGUF)                                            | Sabiá-7B is a Portuguese auto-regressive language model based on LLaMA-1-7B, pretrained on a large Portuguese dataset, offering high performance in few-shot tasks and generating text, with research-only licensing.                                                                                                                                                                                   |
| Portuguese | OpenHermesV2-PTBR-portuguese-brazil-gguf            | [Link](https://huggingface.co/skoll520/OpenHermesV2-PTBR-portuguese-brazil-gguf)                 | A finetuned version of Mistral 7B trained on diverse GPT-4 generated data, designed for Portuguese, with extensive filtering and transformation for enhanced performance.                                                                                                                                                                                                                               |
| Catalan    | CataLlama-v0.2-Instruct-SFT-DPO-Merged-GGUF         | [Link](https://huggingface.co/catallama/CataLlama-v0.2-Instruct-SFT-DPO-Merged-GGUF)             | An instruction-tuned model optimized with DPO for various NLP tasks in Catalan, including translation, NER, summarization, and sentiment analysis, built on an auto-regressive transformer architecture.                                                                                                                                                                                                |

> The models listed are suggestions. The best model for your use case will depend on your specific requirements such as
> language, task complexity, and performance needs.
