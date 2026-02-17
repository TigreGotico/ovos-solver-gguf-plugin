from typing import Dict, Optional

from ovos_plugin_manager.templates.agents import SummarizerEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_gguf_plugin.chat import GGUFChatEngine, Llama



class GGUFSummarizer(SummarizerEngine):
    TEMPLATE = """Your task is to summarize the text into a suitable format.
Answer in plaintext with no formatting, 2 paragraphs long at most. 
Focus on the most important information.
---------------------
{content}
"""
    def __init__(self, config: Optional[Dict] = None,
                 gguf_engine: Optional[Llama] = None):
        """
                 Initialize the GGUFSummarizer, apply default settings, and create the internal GGUF chat engine.
                 
                 If `config` does not include a `system_prompt`, a default summarization prompt is set. The internal GGUFChatEngine is constructed and assigned to `self.api`, and `self.prompt_template` is taken from `config["prompt_template"]` if present or falls back to the class `TEMPLATE`.
                 
                 Parameters:
                     config (Optional[Dict]): Configuration options for the summarizer and underlying engine; may include `system_prompt` and `prompt_template`.
                     gguf_engine (Optional[Llama]): Optional pre-initialized Llama engine to pass to the GGUFChatEngine.
                 """
                 super().__init__(config=config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = "Your task is to summarize text in a couple paragraphs."
        self.api = GGUFChatEngine(config=self.config, gguf_engine=gguf_engine)
        self.prompt_template = self.config.get("prompt_template") or self.TEMPLATE

    @property
    def system_prompt(self):
        """
        Current system prompt for the chat engine.
        
        Returns:
            str: The system prompt text.
        """
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        """
        Set the system prompt used by the internal GGUF chat engine.
        
        Parameters:
            value (str): Prompt text providing high-level instructions for the model (e.g., summarization guidance). This replaces the current system prompt.
        """
        self.api.system_prompt = value

    def summarize(self, document: str, lang: Optional[str] = None) -> str:
        """
        Create a concise summary (up to two short paragraphs) of the provided document using the configured prompt template.
        
        Parameters:
        	lang (str, optional): Language code of the document to inform the summarization model when provided.
        
        Returns:
        	str: The generated summary of the document.
        """
        prompt = self.prompt_template.format(content=document)
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ]).content