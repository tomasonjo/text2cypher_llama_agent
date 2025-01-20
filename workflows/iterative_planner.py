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
from llama_index.graph_stores.neo4j import CypherQueryCorrector

from workflows.shared.local_fewshot_manager import LocalFewshotManager
from workflows.shared.sse_event import SseEvent
from workflows.steps.iterative_planner import (
    correct_cypher_step,
    generate_cypher_step,
    get_final_answer_prompt,
    guardrails_step,
    information_check_step,
    initial_plan_step,
    validate_cypher_step,
)

MAX_INFORMATION_CHECKS = 3
MAX_CORRECT_STEPS = 1


class InitialPlan(Event):
    question: str


class GenerateCypher(Event):
    subquery: str
    retries: int


class ValidateCypher(Event):
    subquery: str
    generated_cypher: str
    retries: int


class CorrectCypher(Event):
    cypher: str
    subquery: str
    errors: list[str]
    retries: int


class ExecuteCypher(Event):
    validated_cypher: str
    subquery: str


class InformationCheck(Event):
    cypher: str
    subquery: str
    database_output: list


class FinalAnswer(Event):
    context: str


class IterativePlanningFlow(Workflow):
    def __init__(self, llm, db, embed_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.graph_store = db["graph_store"]
        self.cypher_query_corrector = CypherQueryCorrector(db["corrector_schema"])
        self.few_shot_retriever = LocalFewshotManager()
        self.db_name = db["name"]

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> InitialPlan | FinalAnswer:
        original_question = ev.input
        # Init global vars
        await ctx.set(
            "original_question", original_question
        )  # So we don't need to pass the question all the time
        await ctx.set(
            "subqueries_cypher_history", {}
        )  # History of which queries were executed

        # LLM call
        guardrails_output = await guardrails_step(self.llm, original_question)
        if guardrails_output.get("next_event") == "generate_final_answer":
            context = "The question is not about movies or cast, so I cannot answer the question"
            final_answer = FinalAnswer(context=context)
            return final_answer

        return InitialPlan(question=original_question)

    @step
    async def initial_plan(self, ctx: Context, ev: InitialPlan) -> GenerateCypher:
        original_question = ev.question
        # store in global context
        initial_plan_output = await initial_plan_step(self.llm, original_question)
        subqueries = initial_plan_output["arguments"].get("plan")

        ctx.write_event_to_stream(
            SseEvent(message=f"Plan:{subqueries}", label="Planning")
        )
        await ctx.set(
            "information_checks", 0
        )  # Current number of information check steps
        await ctx.set("dynamic_notebook", "")  # Current knowledge

        # we use this in ctx.collect() to know how many events to wait for
        await ctx.set("count_of_subqueries", len(subqueries[0]))
        await ctx.set("plan", subqueries)

        # Send events in the current step of the plan
        for subquery in subqueries[0]:
            ctx.send_event(GenerateCypher(subquery=subquery, retries=MAX_CORRECT_STEPS))

    @step(num_workers=4)
    async def generate_cypher_step(
        self,
        ctx: Context,
        ev: GenerateCypher,
    ) -> ValidateCypher:
        fewshot_examples = self.fewshot_retriever.get_fewshot_examples(
            ev.subquery, self.db_name
        )

        generated_cypher = await generate_cypher_step(
            self.llm,
            self.graph_store,
            ev.subquery,
            fewshot_examples,
        )

        return ValidateCypher(
            subquery=ev.subquery, generated_cypher=generated_cypher, retries=ev.retries
        )

    @step(num_workers=4)
    async def validate_cypher_step(
        self, ctx: Context, ev: ValidateCypher
    ) -> ExecuteCypher | CorrectCypher:
        results = await validate_cypher_step(
            llm=self.llm,
            graph_store=self.graph_store,
            question=ev.subquery,
            cypher=ev.generated_cypher,
            cypher_query_corrector=self.cypher_query_corrector,
        )
        # if results["next_action"] == "end":  # DB value mapping
        #    return FinalAnswer(context=str(results["mapping_errors"]))
        if results["next_action"] == "execute_cypher":
            return ExecuteCypher(
                subquery=ev.subquery, validated_cypher=ev.generated_cypher
            )
        if results["next_action"] == "correct_cypher" and ev.retries > 0:
            return CorrectCypher(
                subquery=ev.subquery,
                cypher=ev.generated_cypher,
                errors=results["cypher_errors"],
                retries=ev.retries - 1,
            )
        else:  # What to do if no retries left
            # We just run execute cypher and expect an error
            # Improve
            return ExecuteCypher(
                subquery=ev.subquery, validated_cypher=ev.generated_cypher
            )

    @step(num_workers=4)
    async def correct_cypher_step(
        self, ctx: Context, ev: CorrectCypher
    ) -> ValidateCypher:
        ctx.write_event_to_stream(
            SseEvent(
                message=f"Corecting Cypher query: {ev.cypher} due to error: {ev.errors}",
                label=f"Cypher correction: {ev.subquery}",
            )
        )

        results = await correct_cypher_step(
            self.llm,
            self.graph_store,
            ev.subquery,
            ev.cypher,
            ev.errors,
        )

        return ValidateCypher(
            subquery=ev.subquery, generated_cypher=results, retries=ev.retries
        )

    @step
    async def execute_cypher_step(
        self, ctx: Context, ev: ExecuteCypher
    ) -> InformationCheck:
        ctx.write_event_to_stream(
            SseEvent(
                message=f"Executing Cypher query: {ev.validated_cypher}",
                label=f"Cypher Execution: {ev.subquery}",
            )
        )

        try:
            database_output = self.graph_store.structured_query(ev.validated_cypher)[
                :100
            ]  # Hard limit of 100 results
        except Exception as e:  # Dividing by zero, etc... or timeout
            database_output = [e]

        return InformationCheck(
            subquery=ev.subquery,
            cypher=ev.validated_cypher,
            database_output=database_output,
        )

    @step
    async def information_check_step(
        self, ctx: Context, ev: InformationCheck
    ) -> GenerateCypher | FinalAnswer:
        # retrieve from context
        number_of_subqueries = await ctx.get("count_of_subqueries")

        # wait until we receive all events
        result = ctx.collect_events(ev, [InformationCheck] * number_of_subqueries)
        if result is None:
            return None

        # Add executed cypher statements to global state
        subqueries_cypher_history = await ctx.get("subqueries_cypher_history")
        new_subqueries_cypher = {
            item.subquery: {
                "cypher": item.cypher,
                "database_output": item.database_output,
            }
            for item in result
        }

        await ctx.set(
            "subqueries_cypher_history",
            {**subqueries_cypher_history, **new_subqueries_cypher},
        )

        original_question = await ctx.get("original_question")
        dynamic_notebook = await ctx.get("dynamic_notebook")
        plan = await ctx.get("plan")

        # Do the information check
        data = await information_check_step(
            self.llm, result, original_question, dynamic_notebook, plan
        )

        # Get count of information checks done
        information_checks = await ctx.get("information_checks")

        # Go fetch additional information if needed
        if data.get("modified_plan") and information_checks < MAX_INFORMATION_CHECKS:
            ctx.write_event_to_stream(
                SseEvent(
                    message=f"Modified plan: {data.get('modified_plan')}",
                    label="Modified plan",
                )
            )
            await ctx.set(
                "count_of_subqueries", len(data["modified_plan"][0])
            )  # this is used for ctx.collect()
            await ctx.set("dynamic_notebook", data["dynamic_notebook"])
            await ctx.set("plan", data.get("modified_plan")[1:])
            await ctx.set("information_checks", information_checks + 1)
            for subquery in data["modified_plan"][0]:
                ctx.send_event(
                    GenerateCypher(subquery=subquery, retries=MAX_CORRECT_STEPS)
                )
        else:
            return FinalAnswer(context=data["dynamic_notebook"])

    @step
    async def final_answer(self, ctx: Context, ev: FinalAnswer) -> StopEvent:
        original_question = await ctx.get("original_question")
        final_answer_prompt = get_final_answer_prompt()

        # wait until we receive all events
        gen = await self.llm.astream_chat(
            final_answer_prompt.format_messages(
                context=ev.context, question=original_question
            )
        )

        final_answer = ""
        async for response in gen:
            final_answer += response.delta
            ctx.write_event_to_stream(
                SseEvent(message=response.delta, label="Final answer")
            )

        return StopEvent(result={"answer": final_answer, "question": original_question})
