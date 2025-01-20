from typing import List, Optional

from llama_index.core import ChatPromptTemplate
from pydantic import BaseModel, Field

INFORMATION_CHECK_SYSTEM_TEMPLATE = """You are an expert assistant that evaluates whether a set of subqueries, their results, any existing condensed information, and the current query plan provide enough details to answer a given question. Your task is to:

1. Analyze if the available information is sufficient to answer the original question: "{original_question}".
2. Review the remaining steps in the query plan (if any) to determine if they:
   - Should be retained as is.
   - Need modification to address gaps.
   - Can be skipped because the required information is already available.
   - Should be reorganized for optimized execution (e.g., parallel processing).

### Process:

#### 1. Analyze the Original Question
   - Identify the key pieces of information required to fully answer the question.
   - Break the question into smaller components if it involves multiple aspects.

#### 2. Review Available Information
   - Examine the subqueries, their results, and any provided condensed information.
   - Assess if they collectively address all components of the question.

#### 3. Identify Information Gaps
   - Compare the requirements from the question against the available information.
   - Highlight any missing details or incomplete data that must be retrieved to form a complete answer.
   - If any critical subqueries fail to produce results essential for answering the question:
     - Document the missing information in the **dynamic notebook**.
     - Mark the task as **unsolvable** due to the missing data.
     - Do not produce a modified query plan, as the question cannot be answered with the available or fetchable information.

#### 4. Update and Refine the **Dynamic Notebook**
   - Treat the condensed information as a **central knowledge base**:
     - Continuously update it with key details from subquery results.
     - Integrate new data to close gaps and establish connections between facts.
   - Ensure the notebook reflects the current state of knowledge, including any identified gaps or failed subqueries.
   - Document explicitly if the task is unsolvable due to missing information.

#### 5. Modify the Query Plan (If Solvable)
   - If sufficient information is available:
     - Generate a concise and accurate answer to the question.
     - Specify which remaining steps (if any) in the query plan can be skipped.
   - If information is insufficient but fetchable:
     - Suggest additional subqueries to retrieve the missing details.
     - Ensure new subqueries are designed specifically to fill identified gaps.
     - Organize all subqueries into parallel-executable groups wherever possible.
     - Maintain sequential steps only when strict data dependencies exist.
   - If critical gaps exist that cannot be resolved due to failed subqueries, do not modify the query plan and clearly state why the task cannot be completed.

### Key Guidelines:
- **Focus Only on Information Retrieval**: Limit query plans to fetching data and avoid reasoning/analysis tasks.
- **Optimize for Parallel Execution**: Group queries into parallelizable blocks to reduce execution time.
- **Maintain Sequential Order Only When Necessary**: Use sequential steps only when results of one query depend on another.
- **Centralize Knowledge**:
   - Use the dynamic notebook to consolidate all available information.
   - Ensure it remains the authoritative source for answering the question and guiding further steps.
   - Explicitly document unsolvable tasks if critical information is missing.
"""

INFORMATION_CHECK_USER_TEMPLATE = """
Subqueries and their results:
{subqueries}
Existing dynamic notebook:
{dynamic_notebook}
Current remaining plan (if any):
{plan}
Original question: {question}
"""


class IFOutput(BaseModel):
    """
    Represents the output of an information sufficiency evaluation process.
    Contains either a condensed summary of the available information or additional subqueries needed to answer the original question.
    """

    dynamic_notebook: str = Field(
        description="A continuously updated and refined summary integrating subquery results and condensed information. Serves as the central knowledge base to address the original question and guide further subqueries if necessary."
    )
    modified_plan: Optional[List[List[str]]] = Field(
        description="Modified version of the remaining plan steps. Each group contains queries that can be executed in parallel. Null if no remaining plan exists, all gaps have been addressed, or the task is unsolvable due to missing critical information."
    )


def format_subqueries_for_prompt(information_checks: list) -> str:
    """
    Converts a list of InformationCheck objects into a string that can be added to a prompt.

    Args:
        information_checks (List[InformationCheck]): List of information checks to process.

    Returns:
        str: A formatted string representing subqueries and their results.
    """
    subqueries_and_results = []

    for check in information_checks:
        # Extract the first result if available, otherwise use "No result available."
        result = (
            check.database_output[0]
            if check.database_output
            else "No result available."
        )
        subqueries_and_results.append(
            f"- Subquery: {check.subquery}\n  Result: {result}"
        )

    return "\n".join(subqueries_and_results)


async def information_check_step(
    llm, subquery_events, original_question, dynamic_notebook, plan
):
    information_check_msgs = [
        ("system", INFORMATION_CHECK_SYSTEM_TEMPLATE),
        ("user", INFORMATION_CHECK_USER_TEMPLATE),
    ]

    information_check_prompt = ChatPromptTemplate.from_messages(information_check_msgs)

    subqueries = format_subqueries_for_prompt(subquery_events)

    llm_output = await llm.as_structured_llm(IFOutput).acomplete(
        information_check_prompt.format(
            subqueries=subqueries,
            original_question=original_question,
            dynamic_notebook=dynamic_notebook,
            plan=plan,
        )
    )
    llm_output = llm_output.raw

    return {
        "dynamic_notebook": llm_output.dynamic_notebook,
        "modified_plan": llm_output.modified_plan,
    }
