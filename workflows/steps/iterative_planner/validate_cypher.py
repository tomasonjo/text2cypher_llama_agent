from typing import List, Optional

from llama_index.core import ChatPromptTemplate
from neo4j.exceptions import CypherSyntaxError
from pydantic import BaseModel, Field

VALIDATE_CYPHER_SYSTEM_TEMPLATE = """You are a specialized parser focused on analyzing Cypher query statements to extract node property filters. Your task is to identify and extract properties used in WHERE clauses and pattern matching conditions, but only when they contain explicit literal values.

For each Cypher statement, you should:

1. Identify node labels and their associated property filters
2. Extract the property key and its matching literal value
3. Format the output as a JSON object containing a "filters" array with objects having:
   - node_label: The label of the node (e.g., "Person")
   - property_key: The property name being filtered (e.g., "age")
   - property_value: The literal value being matched (e.g., 30)

Rules for extraction:
- Only extract filters that match against literal values (strings, numbers, booleans)
- Include property filters in MATCH patterns (e.g., (p:Person {name: 'John'}))
- Include property filters in WHERE clauses with literal values (e.g., WHERE p.age > 30)
- Handle both simple equality and comparison operators with literal values
- Ignore property-to-property comparisons (e.g., WHERE m1.rating = m2.rating)
- Ignore variable references or dynamic values
- Valid literal values are:
  * Quoted strings (e.g., 'John', "London")
  * Numbers (e.g., 30, 42.5)
  * Booleans (true, false)

Example input 1:
MATCH (p:Person {name: 'John'})-[:KNOWS]->(f:Friend)
WHERE p.age > 30 AND f.city = 'London' AND f.salary = m.salary

Example output 1:
{
    "filters": [
        {
            "node_label": "Person",
            "property_key": "name",
            "property_value": "John"
        },
        {
            "node_label": "Person",
            "property_key": "age",
            "property_value": 30
        },
        {
            "node_label": "Friend",
            "property_key": "city",
            "property_value": "London"
        }
    ]
}

Example input 2:
MATCH (m1:Movie {title: 'Matrix'}), (m2:Movie)
WHERE m1.rating > m2.rating AND m1.year = 1999

Example output 2:
{
    "filters": [
        {
            "node_label": "Movie",
            "property_key": "title",
            "property_value": "Matrix"
        },
        {
            "node_label": "Movie",
            "property_key": "year",
            "property_value": 1999
        }
    ]
}

Note how property-to-property comparisons (f.salary = m.salary, m1.rating > m2.rating) are ignored in the output."""

VALIDATE_CYPHER_USER_TEMPLATE = """Cypher statement: {cypher}"""


class Property(BaseModel):
    """
    Represents a filter condition based on a specific node property in a graph in a Cypher statement.
    """

    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against."
    )


class ValidateCypherOutput(BaseModel):
    """
    Represents the applied filters of a Cypher query's output.
    """

    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )


async def validate_cypher_step(
    llm,
    graph_store,
    question,
    cypher,
    cypher_query_corrector,
):
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    errors = []
    mapping_errors = []

    # Check for syntax errors
    try:
        graph_store.structured_query(f"EXPLAIN {cypher}")
    except CypherSyntaxError as e:
        errors.append(e.message)

    # Experimental feature for correcting relationship directions
    corrected_cypher = cypher_query_corrector(cypher)
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")

    # Skip mapping the values to the database, LLMs struggle with this tool output
    """
    # Use LLM for mapping for values
    validate_cypher_msgs = [
        ("system", VALIDATE_CYPHER_SYSTEM_TEMPLATE),
        ("user", VALIDATE_CYPHER_USER_TEMPLATE),
    ]
    validate_cypher_prompt = ChatPromptTemplate.from_messages(validate_cypher_msgs)
    llm_output = await llm.as_structured_llm(ValidateCypherOutput).acomplete(
        validate_cypher_prompt.format(cypher=cypher)
    )
    llm_output = llm_output.raw

    if llm_output.filters:
        for filter in llm_output.filters:
            # Do mapping only for string values
            try:
                if (
                    not [
                        prop
                        for prop in graph_store.get_schema()["node_props"][
                            filter.node_label
                        ]
                        if prop["property"] == filter.property_key
                    ][0]["type"]
                    == "STRING"
                ):
                    continue
            except:  # if property is hallucinated/doesn't exist in the schema # ToDo handle it better
                continue
            mapping = graph_store.structured_query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                {"value": filter.property_value},
            )
            if not mapping:
                mapping_errors.append(
                    f"Could not find node in graph with label '{filter.node_label}' where property '{filter.property_key}' equals '{filter.property_value}'. "
                    f"Without this information, I cannot provide a complete answer to your question. "
                    f"If you meant something else, please rephrase your question or verify the specific {filter.property_key} you're asking about. "
                    f"Would you like to try with a different {filter.property_key} value?"
                )

    if mapping_errors:
        next_action = "end"
    """
    if errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "mapping_errors": mapping_errors,
        "steps": ["validate_cypher"],
    }
