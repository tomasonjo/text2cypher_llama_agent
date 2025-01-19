import json

from typing import Any, Type
from fastapi import Request
from jinja2 import pass_context
from llama_index.core.workflow import Workflow

from app.resource_manager import ResourceManager
from app.settings import WORKFLOW_MAP


resource_manager = ResourceManager()
resource_manager.init_llms()
resource_manager.init_databases()
resource_manager.init_embed_model()


# force HTTPS in jinja templates
@pass_context
def urlx_for(
    context: dict,
    name: str,
    **path_params: Any,
) -> str:
    request: Request = context["request"]
    http_url = request.url_for(name, **path_params)
    if scheme := request.headers.get("x-forwarded-proto"):
        return http_url.replace(scheme=scheme)
    return http_url


# run workflow that is present in app/workflows folder
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
