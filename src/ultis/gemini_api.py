from pydantic import BaseModel
from google import genai
import os
import dotenv
import time
import enum

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")



def gemini_qa(msg: str, retries: int = 5) -> str:
    response = "Empty"
    for i in range(retries):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=msg,
            )
            return response.text
        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
    return response

def gemini_qa_ideal_exact_answer(msg: str,  retries: int = 5) -> tuple[str, str]:
    response = "Empty"

    class Answer(BaseModel):
        ideal_answer: str
        exact_answer: list[str]

    for i in range(retries):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=msg,
                # temperature=temperature,
                config={'response_mime_type': 'application/json',
                        'response_schema': Answer}
            )
            response = response.parsed # type: Answer
            ideal_answer = response.ideal_answer
            exact_answer = response.exact_answer
            return ideal_answer, exact_answer

        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)
    return "", ""

def gemini_query_expansion(query: str, retries: int = 5) -> str:
    response = "Empty"
    class Concept(BaseModel):
        name: str
        expanded_terms: str
    
    class COT(BaseModel):
        concepts: list[Concept]
        final_answer: str
    
    for i in range(retries):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""
                    You are an assistant that helps expand search queries with related terms and synonyms in the biomedical domain.
                    Convert the query {query} into a Boolean AND/OR format with query expansion. Follow these steps:
                        1. Analyze the query into [X] main concepts, prioritizing named entities or core biomedical terms (e.g., drugs, diseases, proteins) over descriptive aspects (e.g., mechanism, effect).
                        2. Expand each concept with up to 3 synonyms (including the original term):
                            If no suitable synonyms are found, use only the original term.
                            If only 1-2 suitable synonyms are found, include them.
                        3. Combine synonyms within the same concept using OR.
                        4. Combine the concepts using AND.
                    """
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={'response_mime_type': 'application/json',
                        'response_schema': COT}
            )
            response = response.parsed # type: COT
            return response
        except Exception as e:
            print(f"Attempt qe {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)
    return response

def gemini_classification(question: str, retries: int = 5) -> str:
    msg = f"""
    You are an expert in biomedical question classification. Given a biomedical question, determine whether it requires a single-hop or multi-hop reasoning to answer.
    Classify the question "{question}" as either 'single' or 'multi'
    """
    class Classification(enum.Enum):
        multi = 'multi'
        single = 'single'

    class Question(BaseModel):
        question: str
        classification: str
    response = "Empty"

    for i in range(retries):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=msg,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': Question.model_json_schema(),  # chuyển sang JSON schema
                },
            )
            response = response.parsed # type: Question
            return response["classification"]
        except Exception as e:
            print(f"Attempt classification {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)
    return ""


def gemini_decomposition(question: str, retries: int = 5) -> list[str]:
    msg = f"""
    You are an expert in biomedical question decomposition. Given a complex biomedical question, decompose it into a sequence of simpler sub-questions that can be answered step-by-step to arrive at the final answer.
    Decompose the question "{question}" into a list of simpler sub-questions.
    """
    class Decomposition(BaseModel):
        sub_questions: list[str]
    
    response = "Empty"

    for i in range(retries):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=msg,
                config={'response_mime_type': 'application/json',
                        'response_schema': Decomposition}
            )
            response = response.parsed # type: Decomposition
            return response.sub_questions
        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)
    return []