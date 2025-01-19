from llama_index.core.workflow import Event


class SseEvent(Event):
    label: str
    message: str
