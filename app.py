import os
import requests
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import google.generativeai as genai

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")  # Default to local backend
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini if you want to use local AI calls (optional)
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-flash")

# Function to extract resume text (for local AI processing, optional)
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

# Function to query Gemini locally (optional)
def get_gemini_response(prompt, resume_text, job_desc):
    full_prompt = f"""{prompt}

Resume:
{resume_text}

Job Description:
{job_desc}
"""
    response = model.generate_content(full_prompt)
    return response.text

# Streamlit page setup
st.set_page_config(page_title="ATS Resume Expert", page_icon="🤖")
st.title("🤖 ATS Resume Expert")
st.markdown("Upload your resume and compare with a job description.")

# Input fields
job_desc = st.text_area("📝 Paste Job Description")
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("✅ Resume uploaded successfully!")

# Analyze Resume button
if st.button("📋 Analyze Resume"):
    if not uploaded_file or not job_desc.strip():
        st.warning("⚠️ Please upload a resume and paste a job description.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        data = {"job_desc": job_desc, "do_match": "true"}

        with st.spinner("Analyzing..."):
            try:
                res = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data)
                if res.status_code == 200:
                    result = res.json()
                    st.success("✅ Analysis complete!")

                    st.subheader("📋 Evaluation")
                    st.write(result.get("evaluation", ""))

                    st.subheader("📊 Match Report")
                    st.write(result.get("match_report", ""))

                    st.subheader("📈 TF-IDF Match Score")
                    st.write(f"{round(result.get('tfidf_score', 0) * 100, 2)}%")
                else:
                    st.error(f"Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# Get Match Score button
if st.button("📊 Get Match Score"):
    if not uploaded_file or not job_desc.strip():
        st.warning("⚠️ Please upload a resume and paste a job description.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        data = {"job_desc": job_desc}

        with st.spinner("Checking ATS match..."):
            try:
                res = requests.post(f"{BACKEND_URL}/match_score", files=files, data=data)
                st.write(f"Status Code: {res.status_code}")
                st.write(f"Response Text: {res.text}")

                if res.status_code == 200:
                    result = res.json()
                    st.success("✅ Match score fetched!")

                    st.subheader("📊 Match Report")
                    st.write(result.get("match_report", ""))

                    st.subheader("📈 TF-IDF Match Score")
                    st.write(f"{round(result.get('tfidf_score', 0) * 100, 2)}%")
                else:
                    st.error(f"Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
