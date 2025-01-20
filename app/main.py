import json
from typing import Type

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llama_index.core.workflow import Workflow
from pydantic import BaseModel

from app.resource_manager import ResourceManager
from app.settings import WORKFLOW_MAP
from app.utils import urlx_for

load_dotenv()

templates = Jinja2Templates(directory="app/templates")
templates.env.globals["url_for"] = urlx_for

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

resource_manager = ResourceManager()


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    workflows = list(WORKFLOW_MAP.keys())
    llms_list = [name for name, _ in resource_manager.llms]
    databases_list = list(resource_manager.databases.keys())

    return templates.TemplateResponse(
        request=request,
        name="pages/index.html",
        context={
            "workflows": workflows,
            "llms": llms_list,
            "databases": databases_list,
        },
    )


class WorkflowPayload(BaseModel):
    llm: str
    database: str
    workflow: str
    context: str


@app.post("/workflow/")
async def workflow(payload: WorkflowPayload):
    llm = payload.llm
    database = payload.database
    workflow = payload.workflow
    context_input = payload.context

    try:
        context = json.loads(context_input)
    except json.JSONDecodeError:
        context = {"input": context_input}

    return StreamingResponse(
        run_workflow(llm=llm, database=database, workflow=workflow, context=context),
        media_type="text/event-stream",
    )


# Main workflow runner function
async def run_workflow(llm: str, database: str, workflow: str, context: dict):
    try:
        workflow_class: Type[Workflow] = WORKFLOW_MAP.get(workflow)
        if not workflow_class:
            raise ValueError(f"Workflow '{workflow}' is not recognized.")

        selected_llm = resource_manager.get_model_by_name(llm)
        selected_database = resource_manager.get_database_by_name(database)

        workflow_instance = workflow_class(
            llm=selected_llm,
            db=selected_database,
            embed_model=resource_manager.embed_model,
            timeout=60,
        )

        handler = workflow_instance.run(**context)

        async for event in handler.stream_events():
            if type(event).__name__ != "StopEvent":
                event_data = json.dumps(
                    {
                        "event_type": type(event).__name__,
                        "label": event.label,
                        "message": event.message,
                    }
                )
                yield f"data: {event_data}\n\n"

        result = await handler

        yield f"data: {json.dumps({'result': result})}\n\n"

    except Exception as ex:
        error = json.dumps(
            {
                "event_type": "error",
                "label": "Error",
                "message": f"Failed to run workflow.\n\n{ex}",
            }
        )
        yield f"data: {error}\n\n"
