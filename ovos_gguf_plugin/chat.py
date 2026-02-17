import os
from typing import Dict, Optional, List, Any, Iterable
from sentence_stream import SentenceBoundaryDetector

from llama_cpp import Llama
from ovos_plugin_manager.templates.agents import ChatEngine, AgentMessage, MessageRole
from ovos_utils.log import LOG


class GGUFChatEngine(ChatEngine):
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 gguf_engine: Optional[Llama] = None):
        """
                 Initialize GGUFChatEngine and load or attach a GGUF Llama model.
                 
                 Parameters:
                     config (Optional[Dict[str, Any]]): Engine configuration. Relevant keys:
                         - "model": local path or hub repo identifier (required if `gguf_engine` is not provided).
                         - "n_gpu_layers": number of GPU layers to offload (default 0).
                         - "chat_format": chat formatting option passed to Llama.
                         - "verbose": verbosity flag for model loading (default True).
                         - "remote_filename": filename to use when loading from a hub (default "*Q4_K_M.gguf").
                         - "system_prompt": optional system prompt to inject into conversations.
                         - "allow_system_prompts": whether incoming system messages are allowed (default False).
                     gguf_engine (Optional[Llama]): Pre-instantiated Llama model to use instead of loading from `config`.
                 
                 Side effects:
                     - Assigns the Llama model to `self.model` (either the provided `gguf_engine` or a model loaded from `config`).
                     - Sets `self.system_prompt` and `self.allow_system` from `config`.
                 
                 Raises:
                     ValueError: If no "model" is present in `config` when `gguf_engine` is not provided.
                 """
                 config = config or {}
        super().__init__(config)
        if gguf_engine:
            self.model = gguf_engine
        else:
            if "model" not in self.config:
                raise ValueError("no 'model' set in config")
            model = self.config["model"]
            if os.path.isfile(model):  # local path
                LOG.info(f"Loading GGUF model: {model}")
                self.model = Llama(
                    model_path=model,
                    n_gpu_layers=self.config.get("n_gpu_layers", 0),
                    chat_format=self.config.get("chat_format"),
                    verbose=self.config.get("verbose", True))
            else:
                fname = self.config.get("remote_filename", "*Q4_K_M.gguf")
                LOG.info(f"Loading GGUF model from hub: {model} from file: {fname}")
                self.model = Llama.from_pretrained(
                    repo_id=model,
                    filename=fname,
                    n_gpu_layers=self.config.get("n_gpu_layers", 0),
                    chat_format=self.config.get("chat_format"),
                    verbose=self.config.get("verbose", True)
                )
            LOG.info("GGUF model loaded!")
        self.system_prompt = self.config.get("system_prompt")
        self.allow_system = self.config.get("allow_system_prompts") or False

    def validate_messages(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        """
        Enforce configured system-prompt rules and return a message list suitable for the model.
        
        Processes the provided message history by optionally removing existing system messages, injecting the configured `system_prompt`, and merging or replacing a leading system message depending on `allow_system`. If the resulting history is empty and a `system_prompt` is configured, returns a single SYSTEM AgentMessage containing that prompt.
        
        Parameters:
            messages (List[AgentMessage]): The input chat history to validate and normalize.
        
        Returns:
            List[AgentMessage]: The processed list of AgentMessage objects ready for the model.
        """
        if not self.allow_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if not messages:
            if self.system_prompt:
                return [AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)]
            return []

        if self.system_prompt:
            sysm = AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            if messages and messages[0].role == MessageRole.SYSTEM:
                if self.allow_system:  # merge system prompts
                    sysm = AgentMessage(role=MessageRole.SYSTEM,
                                        content=self.system_prompt + "\n" + messages[0].content)
                # replace system prompt
                messages[0] = sysm
            else:
                messages.insert(0, sysm)
        return messages

    ###########################################################
    # abstract methods to be implemented by individual plugins
    ###########################################################
    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        """
                      Generate an assistant response for the given conversation history.
                      
                      Parameters:
                          messages (List[AgentMessage]): Conversation messages to base the response on.
                          session_id (str): Session identifier used for tracking context.
                          lang (str, optional): BCP-47 language hint for the response.
                          units (str, optional): Preferred measurement system for the response (e.g., "metric", "imperial").
                      
                      Returns:
                          AgentMessage: An AgentMessage with role ASSISTANT containing the model-generated reply.
                      """
        ans = self.model.create_chat_completion(
            messages=[
                {"role": m.role, "content": m.content}
                for m in self.validate_messages(messages)
            ],
            max_tokens=self.config.get("max_tokens"),
            stream=False
        )["choices"][0]["message"]
        return AgentMessage(role=MessageRole.ASSISTANT, content=ans["content"])

    def stream_tokens(self, messages: List[AgentMessage],
                    session_id: str = "default",
                    lang: Optional[str] = None,
                    units: Optional[str] = None) -> Iterable[str]:
        """
                    Stream generated text tokens from the model as they arrive.
                    
                    Yields string fragments produced by the model's streaming chat completion; fragments may be partial sentences and are not guaranteed to be suitable for direct TTS. When concatenated in order, the fragments form the assistant's full response content.
                    
                    Parameters:
                        messages (List[AgentMessage]): Conversation messages to validate and send to the model.
                        session_id (str): Session identifier.
                        lang (str, optional): Language code.
                        units (str, optional): Unit system.
                    
                    Returns:
                        Iterable[str]: An iterator yielding token/content deltas (string fragments) as they are produced by the model.
                    """
        # With stream=True, the output is of type `Iterator[CompletionChunk]`.
        ans = self.model.create_chat_completion(
            messages=[
                {"role": m.role, "content": m.content}
                for m in self.validate_messages(messages)
            ],
            max_tokens=self.config.get("max_tokens"),
            stream=True
        )
        for item in ans:
            chunk = item['choices'][0]["delta"].get("content")
            if chunk:
                yield chunk

    def stream_sentences(self, messages: List[AgentMessage],
                    session_id: str = "default",
                    lang: Optional[str] = None,
                    units: Optional[str] = None) -> Iterable[str]:
        """
                    Yield complete sentences from the assistant response as they are generated.
                    
                    Assembles and yields full sentences suitable for text-to-speech from the provided conversation messages.
                    
                    Parameters:
                        messages (List[AgentMessage]): Conversation messages to send to the model.
                        session_id (str): Session identifier (optional context for implementations).
                        lang (str, optional): Language code hint for downstream consumers.
                        units (str, optional): Unit system hint for downstream consumers.
                    
                    Returns:
                        Iterable[str]: Complete sentence strings from the assistant response, yielded in generation order.
                    """
        boundary_detector = SentenceBoundaryDetector()
        for tok in self.stream_tokens(messages):
            yield from boundary_detector.add_chunk(tok)
        final_text = boundary_detector.finish()
        if final_text:
            yield final_text


if __name__ == "__main__":
    LOG.set_level("DEBUG")

    cfg = {
        "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "remote_filename": "*q8_0.gguf"
    }

    query = """The possibility of alien life in the solar system has been a topic of interest for scientists and astronomers for many years. The search for extraterrestrial life has been a major focus of space exploration, with numerous missions and discoveries made in recent years. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor.
One of the most promising areas for the search for alien life is the moons of Jupiter and Saturn. These moons, such as Europa and Enceladus, are believed to have subsurface oceans that could potentially harbor life. The presence of water, a key ingredient for life as we know it, has been detected on these moons, and there are also indications of other necessary elements such as carbon, nitrogen, and oxygen.
Another area of interest for the search for alien life is the asteroid belt between Mars and Jupiter. This region is home to millions of asteroids, some of which may have the right conditions for life to exist. For example, some asteroids have been found to have water and organic compounds, which are essential for life.
In addition to the moons and asteroids of the solar system, there are also other potential locations for the search for alien life. For example, there are exoplanets, or planets outside of our solar system, that have been discovered in recent years. Some of these exoplanets are believed to be in the habitable zone, which means they are located in the right distance from their star to potentially have liquid water on their surface.
Despite the potential for alien life in the solar system, there are still many uncertainties and unknowns. The search for extraterrestrial life is a complex and multifaceted endeavor that requires a combination of scientific research, technological advancements, and exploration. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor that holds the potential for groundbreaking discoveries in the future."""
    s = GGUFChatEngine(cfg)
    messages = [AgentMessage(role=MessageRole.USER, content=query)]
    for sent in s.stream_sentences(messages):
        print(sent)