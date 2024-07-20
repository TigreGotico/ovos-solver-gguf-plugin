# GGUFSolver README

## Overview

`GGUFSolver` is a question-answering module that utilizes GGUF models to provide responses to user queries. This solver streams utterances for real-time interaction and is built on the `ovos_plugin_manager.templates.solvers.QuestionSolver` framework.

## Features

- Supports loading GGUF models from local files or remote repositories.
- Streams partial responses for improved interactivity.
- Configurable persona and verbosity settings.
- Capable of providing spoken answers.

## Configuration

`GGUFSolver` requires a configuration dictionary. The configuration should at least specify the model to use. Here is an example configuration:

```python
cfg = {
    "model": "TheBloke/notus-7B-v1-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
```

- `model`: The identifier for the model. It can be a local file path or a repository ID for a remote model.
- `remote_filename`: The specific filename to load from a remote repository.
- `chat_format`: (Optional) Chat formatting settings.
- `verbose`: (Optional) Set to `True` for detailed logging.
- `persona`: (Optional) Persona for the system messages. Default is `"You are a helpful assistant who gives short factual answers"`.
- `max_tokens`: (Optional) Maximum tokens for the response. Default is `512`.

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

Use the `stream_utterances` method to stream responses. This is particularly useful for real-time applications such as voice assistants.

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

## Example Models

Here are some example configurations for different models:

### English Models
```python
cfg = {
    "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
    "remote_filename": "*q8_0.gguf"
}
cfg = {
    "model": "RichardErkhov/GritLM_-_GritLM-7B-gguf",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "QuantFactory/falcon-7b-instruct-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "QuantFactory/Samantha-Qwen-2-7B-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "OuteAI/Lite-Mistral-150M-v2-Instruct-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/TowerInstruct-7B-v0.1-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/Dr_Samantha-7B-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/phi-2-orange-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/phi-2-electrical-engineering-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/Unholy-v2-13B-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/notus-7B-v1-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
cfg = {
    "model": "TheBloke/Sonya-7B-GGUF",
    "remote_filename": "*Q4_K_M.gguf"
}
```

### Portuguese Models

```python
    cfg = {
        "model": "RichardErkhov/PORTULAN_-_gervasio-7b-portuguese-ptpt-decoder-gguf",
        "remote_filename": "*Q4_K_M.gguf"
    }
    cfg = {
        "model": "mradermacher/CabraLlama3-8b-GGUF",
        "remote_filename": "*Q4_K_M.gguf"
    }
    cfg = {
        "model": "recogna-nlp/bode-7b-alpaca-pt-br-gguf",
        "remote_filename": "*q4_k_m.gguf"
    }
    cfg = {
        "model": "recogna-nlp/bode-13b-alpaca-pt-br-gguf",
        "remote_filename": "*q4_k_m.gguf"
    }
    cfg = {
        "model": "TheBloke/sabia-7B-GGUF",
        "remote_filename": "*Q4_K_M.gguf"
    }
    cfg = {
        "model": "skoll520/OpenHermesV2-PTBR-portuguese-brazil-gguf",
        "remote_filename": "*Q4_K_M.gguf"
    }
```

### Catalan Models

```python
cfg = {
    "model": "catallama/CataLlama-v0.2-Instruct-SFT-DPO-Merged-GGUF",
    "remote_filename": "*-Q8.gguf"
}
```