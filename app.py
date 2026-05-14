import csv
import os
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from evaluator import check_answer

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "new_dataset.csv")


def _load_question_rows():
    with open(_DATASET_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_QUESTION_ROWS = _load_question_rows()

# Initialize FastAPI app
app = FastAPI(
    title="AI Answer Evaluator",
    description="API for evaluating descriptive answers using AI semantics and keywords.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class AnswerRequest(BaseModel):
    question_id: int
    student_answer: str


@app.get("/random-question")
async def random_question():
    """Return one random question from new_dataset.csv (ID + text only)."""
    if not _QUESTION_ROWS:
        raise HTTPException(status_code=500, detail="Dataset is empty.")
    row = random.choice(_QUESTION_ROWS)
    return {
        "question_id": int(row["ID"]),
        "question": str(row["Question"]).strip(),
        "total": len(_QUESTION_ROWS),
    }


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(index_path)


@app.post("/check")
async def evaluate_answer(request: AnswerRequest):
    """
    Endpoint to evaluate a student's answer.

    check_answer is CPU/GPU-heavy sync code; run it in a thread pool so it does not
    block the event loop (otherwise /random-question and page reloads hang or fail).
    """
    try:
        result = await run_in_threadpool(
            check_answer, request.question_id, request.student_answer
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount(
    "/static",
    StaticFiles(directory=_STATIC_DIR),
    name="static",
)
