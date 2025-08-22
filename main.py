# backend/main.py
import os
import io
import asyncio
from datetime import datetime
from typing import Optional

import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "ats_db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in .env")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure Gemini API key
genai.configure(api_key=GOOGLE_API_KEY)

# Lazy Gemini model initialization
MODEL = None
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = genai.GenerativeModel("models/gemini-1.5-flash")
    return MODEL

# FastAPI app
app = FastAPI(title="ATS Resume Expert API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mongo client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DB_NAME]
resumes_collection = db["resumes"]


class AnalyzeResponse(BaseModel):
    id: str
    evaluation: str
    match_report: Optional[str]
    tfidf_score: Optional[float]
    stored_at: datetime


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join([p.extract_text() or "" for p in pdf.pages])


async def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def build_prompt_analysis(resume_text: str, job_desc: str) -> str:
    return f"""
You are an experienced Technical HR. Evaluate if this resume fits the job role.
Mention key strengths, weaknesses, and concrete suggestions.

Resume:
{resume_text}

Job Description:
{job_desc}
"""


def build_prompt_match(resume_text: str, job_desc: str) -> str:
    return f"""
You are an ATS system.
1. Give a match percentage (0-100).
2. List missing keywords.
3. Provide a short recommendation.

Resume:
{resume_text}

Job Description:
{job_desc}
"""


async def call_gemini(prompt: str) -> str:
    def _call():
        model = get_model()
        resp = model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    return await run_blocking(_call)


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    vect = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vect.fit_transform([text1, text2])
    return float(cosine_similarity(X[0:1], X[1:2])[0][0])


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    file: UploadFile = File(...),
    job_desc: str = Form(...),
    do_match: bool = Form(True)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_bytes = await file.read()
    resume_text = await run_blocking(extract_text_from_pdf_bytes, file_bytes)

    # Save PDF
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = os.path.join(UPLOAD_DIR, f"{ts}_{file.filename}")
    await run_blocking(lambda p, b: open(p, "wb").write(b), path, file_bytes)

    # LLM evaluation
    eval_text = await call_gemini(build_prompt_analysis(resume_text, job_desc))
    match_report = None
    tfidf_score = None

    if do_match:
        match_report = await call_gemini(build_prompt_match(resume_text, job_desc))
        tfidf_score = compute_tfidf_similarity(resume_text, job_desc)

    # Store in DB
    doc = {
        "filename": file.filename,
        "path": path,
        "resume_text": resume_text,
        "job_description": job_desc,
        "evaluation": eval_text,
        "match_report": match_report,
        "tfidf_score": tfidf_score,
        "uploaded_at": datetime.utcnow()
    }
    res = await resumes_collection.insert_one(doc)

    return AnalyzeResponse(
        id=str(res.inserted_id),
        evaluation=eval_text,
        match_report=match_report,
        tfidf_score=tfidf_score,
        stored_at=doc["uploaded_at"]
    )


@app.post("/match_score")
async def match_score(
    file: UploadFile = File(...),
    job_desc: str = Form(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_bytes = await file.read()
    resume_text = await run_blocking(extract_text_from_pdf_bytes, file_bytes)

    # Generate match report using Gemini
    match_report = await call_gemini(build_prompt_match(resume_text, job_desc))
    tfidf_score = compute_tfidf_similarity(resume_text, job_desc)

    return {
        "match_report": match_report,
        "tfidf_score": tfidf_score
    }


@app.get("/")
def root():
    return {"status": "Backend running"}
