from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
import traceback

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# ---------------------------
# CORS CONFIG (allow frontend)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Input Schema
# ---------------------------
class StudyInput(BaseModel):
    subjects: str
    exam_date: str
    hours_per_day: int
    weak_topics: str

# ---------------------------
# System Prompt
# ---------------------------
SYSTEM_PROMPT = (
    "You are an AI Study Planner. Generate a CONCISE, summarized study plan. "
    "Format as: Key topics to study | Daily focus areas | Time allocation per subject | Weak topics focus | Revision strategy. "
    "Use bullet points and keep it SHORT (max 10-15 lines total). "
    "NO long daily breakdowns. Just actionable summary for efficient studying."
)

# ---------------------------
# Groq Setup (Fast & FREE - No Credits Needed)
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set in .env file!")

client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt: str) -> str:
    """Call Groq API using official SDK"""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1000,
            top_p=1,
        )
        
        # SDK returns the message content here for non-stream calls
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/generate-plan")
async def generate_plan(data: StudyInput):
    try:
        prompt = f"""
Subjects: {data.subjects}
Exam Date: {data.exam_date}
Hours per day: {data.hours_per_day}
Weak topics: {data.weak_topics}

Create a detailed daily study plan with specific time slots and revision slots for each topic.
"""
        ai_text = call_groq(prompt)
        return {"plan": ai_text}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to generate plan: {str(e)}"}

@app.get("/")
def root():
    return {"message": "✅ AI Study Planner API is running"}
