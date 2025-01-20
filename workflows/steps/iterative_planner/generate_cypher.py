from llama_index.core import ChatPromptTemplate

GENERATE_SYSTEM_TEMPLATE = """Given an input question, convert it to a Cypher query. No pre-amble.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"""

GENERATE_USER_TEMPLATE = """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""


async def generate_cypher_step(llm, graph_store, subquery, fewshot_examples):
    schema = graph_store.get_schema_str(exclude_types=["Actor", "Director"])

    generate_cypher_msgs = [
        ("system", GENERATE_SYSTEM_TEMPLATE),
        ("user", GENERATE_USER_TEMPLATE),
    ]

    text2cypher_prompt = ChatPromptTemplate.from_messages(generate_cypher_msgs)

    response = await llm.achat(
        text2cypher_prompt.format_messages(
            question=subquery,
            schema=schema,
            fewshot_examples=fewshot_examples,
        )
    )

    return response.message.content
