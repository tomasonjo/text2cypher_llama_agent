import os

from dotenv import load_dotenv
from llama_index.core.workflow import Event
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import (
    CypherQueryCorrector,
    Neo4jPropertyGraphStore,
    Schema,
)
from llama_index.llms.openai import OpenAI

load_dotenv()


class SseEvent(Event):
    label: str
    message: str


default_llm = OpenAI(model="gpt-4o-2024-11-20", temperature=0)

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
graph_store = Neo4jPropertyGraphStore(
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
    url=os.getenv("NEO4J_URI"),
    enhanced_schema=True,
    create_indexes=False,
    timeout=10,
)

if os.getenv("FEWSHOT_NEO4J_USERNAME"):
    fewshot_graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("FEWSHOT_NEO4J_USERNAME"),
        password=os.getenv("FEWSHOT_NEO4J_PASSWORD"),
        url=os.getenv("FEWSHOT_NEO4J_URI"),
        refresh_schema=False,
        create_indexes=False,
        timeout=10,
    )
else:
    fewshot_graph_store = None


def retrieve_fewshots(question):
    if not fewshot_graph_store:
        return
    embedding = embed_model.get_text_embedding(question)
    examples = fewshot_graph_store.structured_query(
        """MATCH (f:Fewshot)
WITH f, vector.similarity.cosine(f.embedding, $embedding) AS score
ORDER BY score DESC LIMIT 7
RETURN f.question AS question, f.cypher AS cypher
""",
        param_map={"embedding": embedding},
    )
    return examples


def store_fewshot_example(question, cypher, llm, success= True):
    if not fewshot_graph_store:
        return
    label = "Fewshot" if success else "Missing"
    # Check if already exists
    already_exists = fewshot_graph_store.structured_query(
        f"MATCH (f:`{label}` {{id: $question + $llm}}) RETURN True",
        param_map={"question": question, 'llm':llm},
    )
    if already_exists:
        return
    # Calculate embedding
    embedding = embed_model.get_text_embedding(question)
    # Store response
    fewshot_graph_store.structured_query(
        f"""MERGE (f:`{label}` {{id: $question + $llm}}) 
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


def check_ok(text):
    # Split the text into words
    words = text.strip().split()

    # Check if there are any words
    if not words:
        return False

    # Check first and last words
    first_word = words[0]
    last_word = words[-1]

    # Return True if either first or last word is "Ok." or "Ok,"
    return first_word in ["Ok.", "Ok"] or last_word in ["Ok.", "Ok"]


# Cypher query corrector is experimental
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph_store.get_schema().get("relationships")
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)

fewshot_examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]
