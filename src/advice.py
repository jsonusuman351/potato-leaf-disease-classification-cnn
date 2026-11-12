"""General, non-diagnostic advice. Optional Gemini improves wording only."""
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ADVICE = {
    "Healthy": (
        "The image resembles a healthy potato leaf. Continue checking plants regularly, "
        "water appropriately, and keep growing conditions suitable."
    ),
    "Early Blight": (
        "The image resembles early blight. Check nearby leaves, reduce prolonged leaf "
        "wetness, and consult a local agricultural specialist before treatment."
    ),
    "Late Blight": (
        "The image resembles late blight. Monitor nearby plants promptly, avoid "
        "unnecessary leaf wetness, and contact a local agricultural specialist."
    ),
}


def get_advice(label: str) -> str:
    fallback = DEFAULT_ADVICE.get(label, "Consult an agricultural specialist for advice.")
    if not os.getenv("GEMINI_API_KEY"):
        return fallback
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        answer = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=(
                "Write exactly 2 short simple sentences with general plant-care precautions "
                "for this CNN-predicted potato-leaf label: " + label + ". "
                "This label is uncertain and not a confirmed diagnosis. "
                "Do not change the label, do not recommend pesticides or doses, "
                "and recommend consulting a local agricultural specialist if concerned."
            ),
        )
        return answer.text.strip() if answer.text else fallback
    except Exception:
        # A missing API key, quota or network should not break prediction.
        return fallback
