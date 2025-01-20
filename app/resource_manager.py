import os

from google.api_core import retry
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore, Schema
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike


class ResourceManager:
    llms = []
    databases = {}
    embed_model = None

    def __init__(self):
        self.init_llms()
        self.init_databases()
        self.init_embed_model()

    def init_llms(self):
        print("> Initializing all llms. This may take some time...")

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

    def init_databases(self):
        print("> Initializing all databases. This may take some time...")
        demo_databases = os.getenv("NEO4J_DEMO_DATABASES")

        if demo_databases != None:
            demo_databases = demo_databases.split(",")
            for db in demo_databases:
                print(f"-> Initializing demo database: {db}")
                try:
                    graph_store = Neo4jPropertyGraphStore(
                        url=os.getenv("NEO4J_URI"),
                        username=db,
                        password=db,
                        database=db,
                        enhanced_schema=True,
                        create_indexes=False,
                        timeout=30,
                    )
                    print(f"-> Getting corrector schema for {db} database.")
                    corrector_schema = self.get_corrector_schema(graph_store)

                    self.databases[db] = {
                        "graph_store": graph_store,
                        "corrector_schema": corrector_schema,
                        "name": db,
                    }
                except Exception as ex:
                    print(ex)

        if os.getenv("NEO4J_DATABASE"):
            self.databases["default"] = {
                "uri": os.getenv("NEO4J_URI"),
                "database": os.getenv("NEO4J_DATABASE"),
                "username": os.getenv("NEO4J_USERNAME"),
                "password": os.getenv("NEO4J_PASSWORD"),
            }

        print(f"Loaded {len(self.databases)} databases.")

    def init_embed_model(self):
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    def get_model_by_name(self, name):
        for model_name, model in self.llms:
            if model_name == name:
                return model
        return None

    def get_database_by_name(self, name: str):
        return self.databases[name]

    def get_corrector_schema(
        self, graph_store: Neo4jPropertyGraphStore
    ) -> list[Schema]:
        corrector_schema = [
            Schema(el["start"], el["type"], el["end"])
            for el in graph_store.get_schema().get("relationships")
        ]

        return corrector_schema
