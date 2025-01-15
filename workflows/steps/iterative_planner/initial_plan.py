from typing import List
from pydantic import BaseModel, Field

from llama_index.core import ChatPromptTemplate


class SubqueriesOutput(BaseModel):
    """Defines the output format for transforming a question into parallel-optimized retrieval steps."""

    plan: List[List[str]] = Field(
        description=(
            """A list of query groups where:
        - Each group (inner list) contains queries that can be executed in parallel
        - Groups are ordered by dependency (earlier groups must be executed before later ones)
        - Each query must be a specific information retrieval request
        - Split into multiple steps only if intermediate results return ≤25 values
        - No reasoning or comparison tasks, only data fetching queries"""
        )
    )


SUBQUERIES_SYSTEM_TEMPLATE = """You are a query planning optimizer. Your task is to break down complex questions into efficient, parallel-optimized retrieval steps. Focus ONLY on information retrieval queries, not analysis or reasoning steps.

Key Requirements:
- Group queries that can be executed in parallel into the same list
- Order groups based on data dependencies
- Include ONLY specific information retrieval queries
- Split into multiple steps ONLY if intermediate query results return ≤25 distinct values
- Exclude reasoning tasks, comparisons, or analysis steps
- Prioritize queries that can be executed first and in parallel

For simple, directly answerable questions, return a single query in a single group.

Example 1:
User: "What was the impact of the 2008 financial crisis on Bank of America's stock price and employee count?"
Assistant: [
    # Group 1 - Single group since we're only looking at one company's metrics
    [
        "What was Bank of America's stock price history and employee count from 2007 to 2009?"
    ]
]

Example 2:
User: "Compare the performance of Tesla's Model 3 with BMW's competing models in terms of range and acceleration."
Assistant: [
    # Group 1 - Basic specs can be fetched in parallel, BMW has <25 competing models
    [
        "What is the Tesla Model 3's EPA range and 0-60 mph acceleration time?",
        "What are the EPA ranges and 0-60 mph acceleration times of BMW models competing with Tesla Model 3?"
    ]
]

Example 3:
User: "List the stock performance of all S&P 500 companies in 2022."
Assistant: [
    # Group 1 - Single query since result set would be >25 values
    [
        "What was the stock performance of all S&P 500 companies in 2022?"
    ]
]

Remember:
- Focus on data retrieval only
- Split into multiple steps only if intermediate results return ≤25 values
- Keep queries combined if results would exceed 25 values
- Maximize parallel execution opportunities when splitting is appropriate
- Maintain necessary sequential ordering
- Keep queries specific and self-contained
- Prioritize independent queries first"""


async def initial_plan_step(llm, question):
    query_decompose_msgs = [
        ("system", SUBQUERIES_SYSTEM_TEMPLATE),
        ("user", "{question}"),
    ]

    subquery_template = ChatPromptTemplate.from_messages(query_decompose_msgs)

    queries_output = await llm.as_structured_llm(SubqueriesOutput).acomplete(
        subquery_template.format(question=question)
    )

    return {
        "next_event": "generate_cypher",
        "arguments": {"plan": queries_output.raw.plan, "question": question},
    }
