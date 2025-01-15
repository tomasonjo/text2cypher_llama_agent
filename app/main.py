import json

from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.settings import WORKFLOW_MAP
from app.utils import llm_utils, run_workflow, urlx_for

load_dotenv()

templates = Jinja2Templates(directory="app/templates")
templates.env.globals["url_for"] = urlx_for

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    workflows = list(WORKFLOW_MAP.keys())
    llms_list = [name for name, _ in llm_utils.llms]

    return templates.TemplateResponse(
        request=request,
        name="pages/index.html",
        context={"workflows": workflows, "llms": llms_list},
    )


class WorkflowPayload(BaseModel):
    llm: str
    workflow: str
    context: str


@app.post("/workflow/")
async def workflow(payload: WorkflowPayload):
    llm = payload.llm
    workflow = payload.workflow
    context_input = payload.context

    try:
        context = json.loads(context_input)
    except json.JSONDecodeError:
        context = {"input": context_input}

    return StreamingResponse(
        run_workflow(llm=llm, workflow=workflow, context=context),
        media_type="text/event-stream",
    )
