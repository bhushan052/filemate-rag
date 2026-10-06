import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


LLM_MODEL = os.getenv(
    "OPENAI_LLM_MODEL"
)


SYSTEM_PROMPT = """
You are a document-based AI assistant.

Answer the user's question using the provided context.

Follow these rules:

1. Use the provided context as the primary source.
2. Do not invent information that is not present in the context.
3. If the answer cannot be found in the context, say:
   "The information is not available in the provided documents."
4. Keep the answer clear and concise.
"""


def generate_answer(
    context,
    query
):

    user_prompt = f"""
CONTEXT:

{context}


QUESTION:

{query}
"""

    response = client.responses.create(

        model=LLM_MODEL,

        instructions=SYSTEM_PROMPT,

        input=user_prompt
    )

    return response.output_text