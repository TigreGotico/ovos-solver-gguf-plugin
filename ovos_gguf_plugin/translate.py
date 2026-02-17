from typing import Optional, Dict, Set

from langcodes import Language
from ovos_config import Configuration
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_utils import classproperty
from ovos_utils.lang import standardize_lang_tag
from ovos_gguf_plugin.chat import GGUFChatEngine, Llama


class GGUFTextTranslator(LanguageTranslator):
    def __init__(self, config: Optional[Dict[str, str]] = None,
                 gguf_engine: Optional[Llama] = None):
        """
                 Initialize the GGUFTextTranslator and create its underlying GGUF chat engine.
                 
                 Sets sensible default configuration values when missing:
                 - "system_prompt": "You are a professional translator. Your task is to translate text"
                 - "model": "TheBloke/TowerInstruct-7B-v0.1-GGUF"
                 - "remote_filename": "*Q4_K_M.gguf"
                 - "n_gpu_layers": -1
                 
                 Parameters:
                     config (Optional[Dict[str, str]]): Optional translator configuration; provided keys override defaults.
                     gguf_engine (Optional[Llama]): Optional pre-initialized GGUF engine instance to pass to the chat engine.
                 """
                 super().__init__(config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = "You are a professional translator. Your task is to translate text"
        if "model" not in self.config:
            self.config["model"] = "TheBloke/TowerInstruct-7B-v0.1-GGUF"
        if "remote_filename" not in self.config:
            self.config["remote_filename"] = "*Q4_K_M.gguf"
        if "n_gpu_layers" not in self.config:
            self.config["n_gpu_layers"] = -1
        self.api = GGUFChatEngine(self.config, gguf_engine=gguf_engine)

    @property
    def system_prompt(self):
        """
        Get the current system prompt used by the underlying GGUF chat engine.
        
        Returns:
            str: The system prompt string configured on the internal chat engine.
        """
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        """
        Set the system-level prompt used by the underlying GGUF chat engine.
        
        Parameters:
            value (str): The system prompt text to apply; may be multi-line and will guide the model's overall behavior.
        """
        self.api.system_prompt = value

    def translate(self, text: str, target: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Translate text into a target language, optionally from a specified source language.
        
        Parameters:
            text (str): Text to be translated.
            target (Optional[str]): Target language code or tag; if None, uses Configuration()["lang"].
            source (Optional[str]): Source language code or tag; if provided, instructs the model to translate from that language.
        
        Returns:
            str: The translated text.
        """
        target = target or Configuration()["lang"]
        tgt = Language.get(target).display_name('en')
        if source:
            src = Language.get(source).display_name('en')
            prompt = f"""Translate the following text from {src} into {tgt}.\n{src}: {text}\n{tgt}: """
        else:
            prompt = f"""Translate the following text into {tgt}.\nOriginal: {text}\nTranslated to {tgt}: """
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ]).content


class GGUFTextLangDetector(LanguageDetector):
    def __init__(self, config: Optional[Dict[str, str]] = None,
                 gguf_engine: Optional[Llama] = None):
        """
                 Initialize the GGUFTextLangDetector, applying default configuration and creating the internal GGUF chat engine.
                 
                 Parameters:
                     config (Optional[Dict[str, str]]): Optional configuration overrides; defaults are applied for missing keys:
                         - 'system_prompt': instruction to respond with a BCP language code only
                         - 'remote_filename': '*Q4_K_M.gguf'
                         - 'n_gpu_layers': -1
                     gguf_engine (Optional[Llama]): Optional underlying GGUF engine instance to pass to the chat engine.
                 
                 """
                 super().__init__(config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = "You are a language guru. Your task is to detect text languages and respond with BCP lang codes. You MUST answer ONLY with a language code"
        if "remote_filename" not in self.config:
            self.config["remote_filename"] = "*Q4_K_M.gguf"
        if "n_gpu_layers" not in self.config:
            self.config["n_gpu_layers"] = -1
        self.api = GGUFChatEngine(self.config, gguf_engine=gguf_engine)

    @property
    def system_prompt(self):
        """
        Get the current system prompt used by the underlying GGUF chat engine.
        
        Returns:
            str: The system prompt string configured on the internal chat engine.
        """
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        """
        Set the system-level prompt used by the underlying GGUF chat engine.
        
        Parameters:
            value (str): The system prompt text to apply; may be multi-line and will guide the model's overall behavior.
        """
        self.api.system_prompt = value

    def detect(self, text):
        """
        Detect the language of the provided text and return its standardized BCP language tag.
        
        Parameters:
            text (str): Text whose language should be detected.
        
        Returns:
            str: Detected language tag in BCP format (e.g., "en", "es-ES").
        """
        return standardize_lang_tag(self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=f"Detect the language of this text: {text}")
        ]).content)

    def detect_probs(self, text):
        """
        Return a language-probability mapping for the given text.
        
        Detects the text's language and returns a dictionary mapping the detected BCP language tag to a probability of 1.0. If no language can be determined, returns an empty dictionary.
        
        Parameters:
            text (str): Text to analyze for language detection.
        
        Returns:
            dict: Mapping of the detected language tag to its probability (e.g., {'en-US': 1.0}), or {} if undetermined.
        """
        l = self.detect(text)
        if l:
            return {l: 1.0}
        return {}

    @classproperty
    def available_languages(cls) -> Set[str]:
        """
        Return the language tags this detector can recognize in its current state.
        
        Override in subclasses to advertise supported languages; values should be canonical language tags (e.g., BCP-47).
        
        Returns:
            Set[str]: A set of supported language tags; empty if none are advertised.
        """
        return set()  # TODO


if __name__ == "__main__":
    cfg = {
        "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "remote_filename": "*q8_0.gguf"
    }
    dt = GGUFTextLangDetector(config=cfg)

    print(dt.detect(
        "The easiest way for anyone to contribute is to help with translations! You can help without any programming knowledge via the translation portal"
    ))
    # en

    cfg = {
        "model": "TheBloke/TowerInstruct-7B-v0.1-GGUF",
        "remote_filename": "*Q4_K_M.gguf"
    }
    tx = GGUFTextTranslator(config=cfg)

    print(tx.translate(
        "The easiest way for anyone to contribute is to help with translations! You can help without any programming knowledge via the translation portal",
        target="es-es"))
    # La forma más sencilla de contribuir es ayudando con las traducciones! Puedes ayudar sin conocimientos de programación a través del portal de traducción


    print(tx.translate(
        "(how|what) is the weather [like] [tomorrow] in {location}",
        target="es-es"))
    # ¿Cómo (o qué) es el clima [como] [mañana] en {ubicación}

    tx.system_prompt = """You are a professional translator. 
    Your task is to translate OpenVoiceOS .intent files.

    RULES:
     - each sentence may contain variables between curly braces -> {variable}
     - the {variable_name} inside should NEVER be modified, only translate surrounding text
     - optional words may be indicated between square brackets -> [optional text]
     - alternative text can be indicated with parenthesis and | -> (this|that)
     - you can modify/add/remove optional/alternative words depending on if they make sense for the target language
     - you can NOT modify/add/remove variables, only change the position inside the sentence
     - if there are multiple variables in a sentence, there MUST be some text between them
    """
    print(tx.translate(
        "(how|what) is the weather [like] [tomorrow] in {location}",
        target="es-es"))
    # ¿Cómo es el tiempo [en] [mañana] en {location}
