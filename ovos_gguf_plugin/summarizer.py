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
        super().__init__(config=config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = "Your task is to summarize text in a couple paragraphs."
        self.api = GGUFChatEngine(config=self.config, gguf_engine=gguf_engine)
        self.prompt_template = self.config.get("prompt_template") or self.TEMPLATE

    @property
    def system_prompt(self):
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.api.system_prompt = value

    def summarize(self, document: str, lang: Optional[str] = None) -> str:
        """
        Create a summary of the provided text.

        Args:
            document (str): The full text to be summarized.
            lang (str, optional): The language of the document.

        Returns:
            str: The summarized text.
        """
        prompt = self.prompt_template.format(content=document)
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ]).content
