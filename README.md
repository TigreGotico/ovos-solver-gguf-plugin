# ovos-gguf-plugin

Unified GGUF wrapper for OpenVoiceOS — chat, summarization, dialog rewriting, translation, language detection, and text embeddings, all backed by quantized GGUF models via `llama-cpp-python`.

## Install

```bash
pip install ovos-gguf-plugin
```

For GPU inference, rebuild `llama-cpp-python` with CUDA support first:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Plugin entry points

| Entry-point group | Plugin name | Class | Role |
|---|---|---|---|
| `opm.agents.chat` | `ovos-chat-gguf-plugin` | `GGUFChatEngine` | conversational chat / question answering |
| `opm.agents.summarizer` | `ovos-summarizer-gguf-plugin` | `GGUFSummarizer` | text summarization |
| `opm.transformer.dialog` | `ovos-dialog-transformer-gguf-plugin` | `GGUFDialogTransformer` | dialog rewriting |
| `opm.lang.translate` | `ovos-translate-gguf-plugin` | `GGUFTextTranslator` | machine translation |
| `opm.lang.detect` | `ovos-lang-detect-gguf-plugin` | `GGUFTextLangDetector` | language detection |
| `opm.embeddings.text` | `ovos-gguf-embeddings-plugin` | `GGUFEmbeddings` | text embeddings |

## Quickstart

### Chat

```python
from ovos_gguf_plugin.chat import GGUFChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

engine = GGUFChatEngine({
    "model": "afrideva/Smol-Llama-101M-Chat-v1-GGUF",
    "remote_filename": "*q2_k.gguf",
    "max_tokens": 128,
})
msgs = [AgentMessage(role=MessageRole.USER, content="Tell me a joke.")]
# stream sentence-by-sentence (suitable for TTS)
for sentence in engine.stream_sentences(msgs):
    print(sentence)
# or get the full response at once
reply = engine.continue_chat(msgs)
print(reply.content)
```

### Summarizer

```python
from ovos_gguf_plugin.summarizer import GGUFSummarizer

s = GGUFSummarizer({
    "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
    "remote_filename": "*q8_0.gguf",
})
print(s.summarize("Long document text goes here ... " * 20))
```

### Dialog transformer

```python
from ovos_gguf_plugin.dialog_transformers import GGUFDialogTransformer

dt = GGUFDialogTransformer({
    "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
    "remote_filename": "*q8_0.gguf",
})
print(dt.transform("gonna grab some food real quick"))
```

### Translation

```python
from ovos_gguf_plugin.translate import GGUFTextTranslator

tx = GGUFTextTranslator({
    "model": "TheBloke/TowerInstruct-7B-v0.1-GGUF",
    "remote_filename": "*Q4_K_M.gguf",
})
print(tx.translate("the easiest way to contribute is to help with translations",
                   target="es-es"))
```

### Language detection

```python
from ovos_gguf_plugin.translate import GGUFTextLangDetector

dt = GGUFTextLangDetector({
    "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
    "remote_filename": "*q8_0.gguf",
})
print(dt.detect("you can help without any programming knowledge"))  # → en
```

### Text embeddings

```python
from ovos_gguf_plugin.embeddings import GGUFEmbeddings

emb = GGUFEmbeddings({"model": "all-MiniLM-L6-v2"})
vector = emb.get_embeddings("hello world")
print(len(vector), "dims")
```

`model` accepts a friendly name from `GGUFEmbeddings.DEFAULT_MODELS` (e.g. `labse`, `all-MiniLM-L6-v2`, `nomic-embed-text-v1.5`, `bge-large-en-v1.5`), a bare Hugging Face repo id (with `remote_filename`), or a local `.gguf` path. Default is `labse`.

As an OVOS text-embeddings plugin it is selected by name (`ovos-gguf-embeddings-plugin`), so it is a drop-in for anything that previously used the standalone embeddings plugin.

## Configuration

All wrappers share the same config keys:

| Key | Default | Description |
|---|---|---|
| `model` | required | Local `.gguf` path, HuggingFace repo id, or friendly name (embeddings) |
| `remote_filename` | `*Q4_K_M.gguf` | Glob for selecting the file from a HF repo |
| `n_gpu_layers` | `0` | GPU layers to offload (`-1` = all) |
| `chat_format` | `None` | llama.cpp chat format (auto-detected for most models) |
| `verbose` | `True` | llama.cpp verbosity |
| `max_tokens` | `512` | Maximum tokens to generate |
| `system_prompt` | locale default | Override the system prompt |

See [`docs/configuration.md`](docs/configuration.md) for the full reference including per-wrapper options and GPU build instructions.

## Localized prompts

System prompts and templates ship as `.prompt` resource files under `ovos_gguf_plugin/locale/<lang>/` and are loaded via [ovos-spec-tools](https://github.com/OpenVoiceOS/ovos-spec-tools) (OVOS-INTENT-2 §4.4). Drop translated `.prompt` files under a new `locale/<lang>/` to add a language; English `en-us` ships by default and is the fallback. A `system_prompt` in config overrides the locale file.

See [`docs/localization.md`](docs/localization.md) for the full guide.

## OVOS Persona Framework

```json
{
  "name": "MyAssistant",
  "solvers": ["ovos-solver-gguf-plugin"],
  "ovos-solver-gguf-plugin": {
    "model": "TheBloke/notus-7B-v1-GGUF",
    "remote_filename": "*Q4_K_M.gguf",
    "persona": "You are a helpful assistant.",
    "verbose": false
  }
}
```

```bash
ovos-persona-server --persona my_persona.json
```

## Documentation

- [`docs/configuration.md`](docs/configuration.md) — full config reference, GPU build, per-wrapper notes
- [`docs/localization.md`](docs/localization.md) — `.prompt` system, adding a language
- [`docs/models.md`](docs/models.md) — recommended models per wrapper (including tiny CI-friendly ones)

## Examples

Runnable scripts under [`examples/`](examples/):

- [`chat_example.py`](examples/chat_example.py)
- [`embeddings_example.py`](examples/embeddings_example.py)
- [`translate_example.py`](examples/translate_example.py)
- [`lang_detect_example.py`](examples/lang_detect_example.py)
- [`summarize_example.py`](examples/summarize_example.py)

## Testing

```bash
pip install "ovos-gguf-plugin[test]"
python -m pytest test/ -v
```

The test suite contains:

- `test/test_embeddings.py` — hermetic unit tests (mocked llama.cpp, no downloads)
- `test/test_prompts.py` — hermetic unit tests for localized prompt loading
- `test/test_e2e.py` — real-model end-to-end tests (downloads tiny GGUFs once, ~70 MB total):
  - chat: `afrideva/Smol-Llama-101M-Chat-v1-GGUF` q2_k (~45 MB)
  - embeddings: `leliuga/all-MiniLM-L6-v2-GGUF` Q4_K_M (~23 MB)

## Credits

Originally developed by [TigreGótico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org),
sponsored by VisioLab. Modernized under the [NGI0 Commons Fund](https://nlnet.nl/commonsfund) / [NLnet](https://nlnet.nl).

<img src="https://github.com/user-attachments/assets/809588a2-32a2-406c-98c0-f88bf7753cb4" width="220" alt="VisioLab"/>

> This work was sponsored by VisioLab, part of [Royal Dutch Visio](https://visio.org/), is the test, education, and research center in the field of (innovative) assistive technology for blind and visually impaired people and professionals. We explore (new) technological developments such as Voice, VR and AI and make the knowledge and expertise we gain available to everyone.

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
