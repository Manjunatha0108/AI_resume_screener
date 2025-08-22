🤖 ATS Resume Expert

An AI-powered Resume Screening and ATS Match Analysis Tool.

🚀 Features

Upload Resume (PDF) – Extracts text using pdfplumber.

AI-Powered Evaluation – Uses Google Gemini to evaluate resume strengths, weaknesses, and suggestions.

ATS Match Report – Generates a match percentage, missing keywords, and recommendations.

TF-IDF Similarity Score – Calculates text similarity between resume and job description using scikit-learn.

Database Storage – Saves resumes and analysis reports in MongoDB.

FastAPI Backend – Handles resume processing, AI calls, and API endpoints.

Streamlit Frontend – Provides an easy-to-use web interface for uploading resumes and viewing results.

🛠️ Tech Stack

Backend: FastAPI, MongoDB, Motor, pdfplumber, scikit-learn

Frontend: Streamlit

AI: Google Gemini (Generative AI API)

Other: dotenv, requests, TF-IDF, cosine similarity

📌 How It Works

Upload your resume (PDF) and paste the job description.

Backend extracts text and analyzes with Gemini.

Returns:

HR-style evaluation

ATS match report

TF-IDF similarity score

Results are displayed on the Streamlit dashboard.
