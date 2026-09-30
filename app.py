import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "chatbot_config.txt"

app = Flask(__name__)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = CONFIG_PATH.read_text(encoding="utf-8").strip()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    prompt = f"{SYSTEM_PROMPT}\n\nUser message:\n{message}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        reply = (response.text or "").strip()

        if not reply:
            reply = "I could not generate a response. Please try again."

        return jsonify({"reply": reply})
    except Exception:
        return jsonify({
            "error": "The chatbot could not process your request right now."
        }), 500


if __name__ == "__main__":
    app.run()
