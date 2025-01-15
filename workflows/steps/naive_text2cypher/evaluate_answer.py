from llama_index.core import ChatPromptTemplate

EVALUATE_ANSWER_SYSTEM_TEMPLATE = """You are a helpful assistant that must determine if the provided database output (from a Cypher query) is sufficient and relevant to answer a given question. You will receive three inputs:
1. question – The user’s question.
2. cypher – The Cypher query used.
3. database_output – The query results.
Your task is:
* Check if the database_output is enough to answer the question meaningfully.
* If sufficient, reply with "Ok".
* If insufficient, explain what’s wrong (e.g., missing data, incorrect query structure, irrelevant results) and how to fix the Cypher query (or approach) so it would produce the necessary answer."""

EVALUATE_ANSWER_USER_TEMPLATE = """You are given the following information:
Question:
{question}
Cypher Query:
{cypher}
Database Output:
{context}
Analyze whether the provided database output is adequate to answer the question.
* If the context is sufficient, return "Ok".
* Otherwise, describe in detail what the error or shortcoming is, and how to correct the Cypher query (or the approach).
"""


async def evaluate_database_output_step(llm, subquery, cypher, context):
    # Correct cypher
    correct_cypher_messages = [
        ("system", EVALUATE_ANSWER_SYSTEM_TEMPLATE),
        ("user", EVALUATE_ANSWER_USER_TEMPLATE),
    ]

    correct_cypher_prompt = ChatPromptTemplate.from_messages(correct_cypher_messages)
    response = await llm.achat(
        correct_cypher_prompt.format_messages(
            question=subquery, cypher=cypher, context=context
        )
    )
    return response.message.content
