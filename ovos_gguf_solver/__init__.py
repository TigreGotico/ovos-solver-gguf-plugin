import os
from typing import Optional, Iterable

import llama_cpp
from ovos_plugin_manager.templates.solvers import QuestionSolver
from ovos_utils.log import LOG


class GGUFSolver(QuestionSolver):
    enable_tx = False
    priority = 60

    def __init__(self, config=None):
        config = config or {}
        super().__init__(config)
        if "model" not in self.config:
            raise ValueError("no 'model' set in config")
        model = self.config["model"]
        if os.path.isfile(model):  # local path
            LOG.info(f"Loading GGUF model: {model}")
            self.model = llama_cpp.Llama(
                model_path=model,
                chat_format=self.config.get("chat_format"),
                verbose=self.config.get("verbose", True))
        else:
            fname = self.config.get("remote_filename", "*Q4_K_M.gguf")
            LOG.info(f"Loading GGUF model from hub: {model} from file: {fname}")
            self.model = llama_cpp.Llama.from_pretrained(
                repo_id=model,
                filename=fname,
                chat_format=self.config.get("chat_format"),
                verbose=self.config.get("verbose", True)
            )

    def stream_utterances(self, query: str,
                          context: Optional[dict] = None) -> Iterable[str]:
        """streaming api, yields utterances as they become available
        each utterance can be sent to TTS before we have a full answer
        this is particularly helpful with LLMs"""
        # With stream=True, the output is of type `Iterator[CompletionChunk]`.
        ans = self.model.create_chat_completion(
            messages=[
                {"role": "system",
                 "content": self.config.get("persona", "You are an helpful assistant who gives short factual answers")},
                {"role": "user",
                 "content": query}
            ],
            max_tokens=self.config.get("max_tokens"),
            stream=True
        )
        # Iterate over the output and print it.
        answer = ""
        for item in ans:
            chunk = item['choices'][0]["delta"].get("content")
            if not chunk:
                continue
            answer += chunk
            if any(chunk.endswith(p) for p in [".", "!", "?", "\n", ":"]):
                if len(chunk) >= 2 and chunk[-2].isdigit() and chunk[-1] == ".":
                    continue  # dont split numbers
                if answer.strip():
                    yield answer.strip()
                answer = ""

    def get_spoken_answer(self, query: str, context: Optional[dict] = None) -> str:
        ans = self.model.create_chat_completion(
            messages=[
                {"role": "system",
                 "content": self.config.get("persona", "You are an helpful assistant who gives short factual answers")},
                {"role": "user",
                 "content": query}
            ],
            max_tokens=self.config.get("max_tokens")
        )
        return ans["choices"][0]["message"]["content"]


if __name__ == "__main__":
    LOG.set_level("DEBUG")

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
    # Catalan models
    cfg = {
        "model": "catallama/CataLlama-v0.2-Instruct-SFT-DPO-Merged-GGUF",
        "remote_filename": "*-Q8.gguf"
    }
    # Portuguese models
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


    cfg = {
        "model": "TheBloke/phi-2-orange-GGUF",
        "remote_filename": "*Q4_K_M.gguf"
    }
    s = GGUFSolver(cfg)
    query = "tell me a joke about aliens"
    for sent in s.stream_utterances(query):
        print(sent)
