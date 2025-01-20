from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from workflows.shared.local_fewshot_manager import LocalFewshotManager
from workflows.shared.neo4j_fewshot_manager import Neo4jFewshotManager
from workflows.shared.sse_event import SseEvent
from workflows.shared.utils import check_ok
from workflows.steps.naive_text2cypher import (
    correct_cypher_step,
    evaluate_database_output_step,
    generate_cypher_step,
    get_naive_final_answer_prompt,
)


class SummarizeEvent(Event):
    question: str
    cypher: str
    context: str
    evaluation: str


class ExecuteCypherEvent(Event):
    question: str
    cypher: str


class CorrectCypherEvent(Event):
    question: str
    cypher: str
    error: str


class EvaluateEvent(Event):
    question: str
    cypher: str
    context: str


class NaiveText2CypherRetryCheckFlow(Workflow):
    max_retries = 2

    def __init__(self, llm, db, embed_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.graph_store = db["graph_store"]
        self.embed_model = embed_model
        self.db_name = db["name"]

        # Fewshot graph store allows for self learning loop by storing new examples
        self.fewshot_manager = Neo4jFewshotManager()
        if self.fewshot_manager.graph_store:
            self.fewshot_retriever = self.fewshot_manager.retrieve_fewshots
        else:
            self.fewshot_retriever = LocalFewshotManager().retrieve_fewshots

    @step
    async def generate_cypher(self, ctx: Context, ev: StartEvent) -> ExecuteCypherEvent:
        # Init global vars
        await ctx.set("retries", 0)

        question = ev.input

        fewshot_examples = self.fewshot_retriever(
            question, self.db_name, self.embed_model
        )

        cypher_query = await generate_cypher_step(
            llm=self.llm,
            graph_store=self.graph_store,
            subquery=question,
            fewshot_examples=fewshot_examples,
        )
        # Return for the next step
        return ExecuteCypherEvent(question=question, cypher=cypher_query)

    @step
    async def execute_query(
        self, ctx: Context, ev: ExecuteCypherEvent
    ) -> EvaluateEvent | CorrectCypherEvent:
        # Get global var
        retries = await ctx.get("retries")

        ctx.write_event_to_stream(
            SseEvent(message=f"Executing Cypher: {ev.cypher}", label="Cypher execution")
        )
        try:
            # Hard limit to 100 records
            database_output = str(self.graph_store.structured_query(ev.cypher)[:100])
        except Exception as e:
            database_output = str(e)
            ctx.write_event_to_stream(
                SseEvent(
                    message=f"Cypher Execution error: {database_output}",
                    label="Cypher execution error",
                )
            )
            # Retry
            if retries < self.max_retries:
                await ctx.set("retries", retries + 1)
                return CorrectCypherEvent(
                    question=ev.question, cypher=ev.cypher, error=database_output
                )
        ctx.write_event_to_stream(
            SseEvent(
                message=f"Database output: {database_output}", label="Database output"
            )
        )
        return EvaluateEvent(
            question=ev.question, cypher=ev.cypher, context=database_output
        )

    @step
    async def evaluate_context(
        self, ctx: Context, ev: EvaluateEvent
    ) -> SummarizeEvent | CorrectCypherEvent:
        # Get global var
        retries = await ctx.get("retries")
        evaluation = await evaluate_database_output_step(
            self.llm, ev.question, ev.cypher, ev.context
        )
        if retries < self.max_retries and not evaluation == "Ok":
            await ctx.set("retries", retries + 1)
            return CorrectCypherEvent(
                question=ev.question, cypher=ev.cypher, error=evaluation
            )
        return SummarizeEvent(
            question=ev.question,
            cypher=ev.cypher,
            context=ev.context,
            evaluation=evaluation,
        )

    @step
    async def correct_cypher_step(
        self, ctx: Context, ev: CorrectCypherEvent
    ) -> ExecuteCypherEvent:
        ctx.write_event_to_stream(
            SseEvent(
                message=f"Error: {ev.error}",
                label="Cypher correction",
            )
        )
        results = await correct_cypher_step(
            self.llm,
            self.graph_store,
            ev.question,
            ev.cypher,
            ev.error,
        )
        return ExecuteCypherEvent(question=ev.question, cypher=results)

    @step
    async def summarize_answer(self, ctx: Context, ev: SummarizeEvent) -> StopEvent:
        retries = await ctx.get("retries")
        # If retry was successful:
        if retries > 0 and check_ok(ev.evaluation):
            # print(f"Learned new example: {ev.question}, {ev.cypher}")
            self.fewshot_manager.store_fewshot_example(
                question=ev.question,
                cypher=ev.cypher,
                llm=self.llm.model,
                embed_model=self.embed_model,
                database=self.db_name,
            )

        naive_final_answer_prompt = get_naive_final_answer_prompt()
        gen = await self.llm.astream_chat(
            naive_final_answer_prompt.format_messages(
                context=ev.context, question=ev.question, cypher_query=ev.cypher
            )
        )
        final_answer = ""
        async for response in gen:
            final_answer += response.delta
            ctx.write_event_to_stream(
                SseEvent(message=response.delta, label="Final answer")
            )

        stop_event = StopEvent(
            result={
                "cypher": ev.cypher,
                "question": ev.question,
                "answer": final_answer,
            }
        )

        # Return the final result
        return stop_event
