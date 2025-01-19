import os
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class FewshotManager:
    graph_store = None

    def __init__(self):
        if os.getenv("FEWSHOT_NEO4J_USERNAME"):
            self.graph_store = Neo4jPropertyGraphStore(
                username=os.getenv("FEWSHOT_NEO4J_USERNAME"),
                password=os.getenv("FEWSHOT_NEO4J_PASSWORD"),
                url=os.getenv("FEWSHOT_NEO4J_URI"),
                refresh_schema=False,
                create_indexes=False,
                timeout=30,
            )

    def retrieve_fewshots(self, question, embed_model):
        if not self.graph_store:
            return

        embedding = embed_model.get_text_embedding(question)
        examples = self.graph_store.structured_query(
            """MATCH (f:Fewshot)
WITH f, vector.similarity.cosine(f.embedding, $embedding) AS score
ORDER BY score DESC LIMIT 7
RETURN f.question AS question, f.cypher AS cypher""",
            param_map={"embedding": embedding},
        )
        return examples

    def store_fewshot_example(self, question, cypher, llm, embed_model):
        if not self.graph_store:
            return

        # Check if already exists
        already_exists = self.graph_store.structured_query(
            "MATCH (f:Fewshot {id: $question + $llm}) RETURN True",
            param_map={"question": question, "llm": llm},
        )
        if already_exists:
            return

        # Calculate embedding
        embedding = embed_model.get_text_embedding(question)

        # Store response
        self.graph_store.structured_query(
            """MERGE (f:Fewshot {id: $question + $llm})
SET f.cypher = $cypher, f.llm = $llm, f.created = datetime(), f.question = $question
WITH f
CALL db.create.setNodeVectorProperty(f,'embedding', $embedding)""",
            param_map={
                "question": question,
                "cypher": cypher,
                "embedding": embedding,
                "llm": llm,
            },
        )
        return
