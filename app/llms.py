import os

from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai_like import OpenAILike
from google.api_core import retry


class LlmUtils:
    llms = []

    def __init__(self):
        print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY") is not None)
        print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY") is not None)
        print("ANTHROPIC_API_KEY:", os.getenv("ANTHROPIC_API_KEY") is not None)
        print("MISTRAL_API_KEY:", os.getenv("MISTRAL_API_KEY") is not None)
        print("DEEPSEEK_API_KEY:", os.getenv("DEEPSEEK_API_KEY") is not None)

        if os.getenv("OPENAI_API_KEY"):
            self.llms.extend(
                [
                    ("gpt-4o", OpenAI(model="gpt-4o", temperature=0)),
                ]
            )

        if os.getenv("GOOGLE_API_KEY"):
            google_retry = dict(
                retry=retry.Retry(initial=0.1, multiplier=2, timeout=61)
            )
            self.llms.extend(
                [
                    (
                        "gemini-1.5-pro",
                        Gemini(
                            model="models/gemini-1.5-pro",
                            temperature=0,
                            request_options=google_retry,
                        ),
                    ),
                    (
                        "gemini-1.5-flash",
                        Gemini(
                            model="models/gemini-1.5-flash",
                            temperature=0,
                            request_options=google_retry,
                        ),
                    ),
                ]
            )

        if os.getenv("ANTHROPIC_API_KEY"):
            self.llms.extend(
                [
                    (
                        "sonnet-3.5",
                        Anthropic(
                            model="claude-3-5-sonnet-latest",
                            max_tokens=8076,
                        ),
                    ),
                    (
                        "haiku-3.5",
                        Anthropic(
                            model="claude-3-5-haiku-latest",
                            max_tokens=8076,
                        ),
                    ),
                ]
            )

        if os.getenv("MISTRAL_API_KEY"):
            self.llms.extend(
                [
                    (
                        "mistral-medium",
                        MistralAI(
                            model="mistral-medium",
                            api_key=os.getenv("MISTRAL_API_KEY"),
                        ),
                    ),
                    (
                        "mistral-large",
                        MistralAI(
                            model="mistral-large-latest",
                            api_key=os.getenv("MISTRAL_API_KEY"),
                        ),
                    ),
                    (
                        "ministral-8b",
                        MistralAI(
                            model="ministral-8b-latest",
                            api_key=os.getenv("MISTRAL_API_KEY"),
                        ),
                    ),
                ]
            )

        if os.getenv("DEEPSEEK_API_KEY"):
            self.llms.extend(
                [
                    (
                        "deepseek-v3",
                        OpenAILike(
                            model="deepseek-chat",
                            api_base="https://api.deepseek.com/beta",
                            api_key=os.getenv("DEEPSEEK_API_KEY"),
                        ),
                    )
                ]
            )

        print(f"Loaded {len(self.llms)} llms.")

    def get_model_by_name(self, name):
        for model_name, model in self.llms:
            if model_name == name:
                return model
        return None
