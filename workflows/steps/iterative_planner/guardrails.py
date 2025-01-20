from typing import Literal

from llama_index.core import ChatPromptTemplate
from pydantic import BaseModel, Field


class Guardrail(BaseModel):
    """Guardrail"""

    decision: Literal["movie", "end"] = Field(
        description="Decision on whether the question is related to movies"
    )


GUARDRAILS_SYSTEM_PROMPT_TEMPLATE = """As an intelligent assistant, your primary objective is to decide whether a given question is related to movies or not.
If the question is related to movies, output "movie". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it refers to any movie, actor, director, film industry,
or related topics. Provide only the specified output: "movie" or "end"."""


async def guardrails_step(llm, question):
    # Refine Prompt
    chat_refine_msgs = [
        ("system", GUARDRAILS_SYSTEM_PROMPT_TEMPLATE),
        ("user", "The question is: {question}"),
    ]

    guardrails_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

    guardrails_output = await llm.as_structured_llm(Guardrail).acomplete(
        guardrails_template.format(question=question)
    )
    guardrails_output = guardrails_output.raw.decision

    if guardrails_output == "end":
        context = "The question is not about movies or their case, so I cannot answer this question"
        return {
            "next_event": "generate_final_answer",
            "arguments": {"context": context, "question": question},
        }

    return {"next_event": "generate_plan", "arguments": {}}
