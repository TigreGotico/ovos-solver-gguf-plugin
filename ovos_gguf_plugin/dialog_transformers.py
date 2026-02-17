from typing import Tuple, Optional

from ovos_plugin_manager.templates.transformers import DialogTransformer
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_gguf_plugin.chat import GGUFChatEngine, Llama


class GGUFDialogTransformer(DialogTransformer):
    def __init__(self, name="ovos-dialog-transformer-openai-plugin", priority=10, config=None,
                 gguf_engine: Optional[Llama] = None):
        """
                 Initialize the dialog transformer and attach a GGUF chat engine.
                 
                 Ensures the configuration contains a "system_prompt" (defaults to
                 "Your task is to rewrite text as if it was spoken by a different character")
                 and creates a GGUFChatEngine using the provided config and optional low-level
                 GGUF model instance.
                 
                 Parameters:
                 	name (str): Identifier for the transformer.
                 	priority (int): Ordering priority for transformer selection.
                 	config (dict, optional): Configuration passed to the transformer and engine; may contain "system_prompt".
                 	gguf_engine (Llama, optional): Optional low-level GGUF model instance to use with GGUFChatEngine.
                 """
        super().__init__(name, priority, config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = "Your task is to rewrite text as if it was spoken by a different character"
        self.api = GGUFChatEngine(config=self.config, gguf_engine=gguf_engine)

    @property
    def system_prompt(self):
        """
        Get the current system prompt used by the underlying GGUF chat engine.
        
        Returns:
            str: The system prompt text.
        """
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        """
        Set the transformer's system prompt used by the underlying GGUF chat engine.
        
        Parameters:
            value (str): The new system prompt text to apply to the engine.
        """
        self.api.system_prompt = value

    def transform(self, dialog: str, context: dict = None) -> Tuple[str, dict]:
        """
        Rewrite a dialog using a character-specific prompt when available.
        
        Parameters:
            dialog: The dialog text to rewrite.
            context: Optional mapping that may include a "prompt" key (the character-specific rewrite instruction) and other metadata (e.g., language). If None, an empty context is used.
        
        Returns:
            A tuple (transformed_dialog, context) where `transformed_dialog` is the rewritten dialog when a prompt is present, or the original dialog otherwise, and `context` is the unchanged context mapping.
        """
        context = context or {}
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if not prompt:
            return dialog, context
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=f"{prompt} : {dialog}")
        ]).content, context