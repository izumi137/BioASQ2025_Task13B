from pydantic import BaseModel
from openai import OpenAI
import os
import dotenv
import time
from ultis.ultis import get_output_token_openai


dotenv.load_dotenv()

# QE_KEY=os.getenv('QE_KEY')
OPENAI_KEY = os.getenv('OPENAI_KEY')



def openai_qa_normal(msg: str, model_name: str = 'gpt-4o-mini', retries: int = 5, temperature: float = 0.1):
    # client = OpenAI(api_key=QA_KEY)

    max_token = get_output_token_openai(msg, 16384, 128000)
    response = "Empty"

    for i in range (retries):
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            completion = client.chat.completions.create(
                model=model_name,
                messages=msg,
                max_tokens = max_token,
                temperature = temperature,
                # max_tokens=100,
            )
            response = completion.choices[0].message.content
            # self.ideal_answer = response
            return response

        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)

    return response

def openai_qa_summary(msg: str, model_name: str = 'gpt-4o-mini', retries: int =5, temperature: float = 0.1):
    # client = OpenAI(api_key=QA_KEY)

    max_token = get_output_token_openai(msg, 16384, 128000)
    response = "Empty"

    for i in range (retries):
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            completion = client.chat.completions.create(
                model=model_name,
                messages=msg,
                max_tokens = max_token,
                temperature = temperature,
                # max_tokens=100,
            )
            response = completion.choices[0].message.content
            # self.ideal_answer = response
            return response

        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)

    return response

def openai_qa_other(msg: str, model_name: str = 'gpt-4o-mini', retries: int =5, temperature: float = 0.1):
    # client = OpenAI(api_key=QA_KEY)

    class Answer(BaseModel):
        ideal_answer: str
        exact_answer: list[str]

    max_token = get_output_token_openai(msg, 16384, 128000)
    response = "Empty"

    for i in range (retries):
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            completion = client.beta.chat.completions.parse(
                model=model_name,
                max_tokens = max_token,
                temperature=temperature,
                messages=msg,
                response_format=Answer,
            )
            response = completion.choices[0].message.parsed
            ideal_answer = response.ideal_answer
            exact_answer = response.exact_answer
            return ideal_answer, exact_answer
        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)

    return "", ""


def openai_query_expansion(query: str, model_name: str = 'gpt-4o-mini', retries: int = 5) -> str:
    # client = OpenAI(QE_KEY)

    class Concept(BaseModel):
        name: str
        expanded_terms: str
    
    class COT(BaseModel):
        concepts: list[Concept]
        final_answer: str
    
    for i in range(retries):
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant that helps expand search queries with related terms and synonyms in the biomedical domain."},
                    {"role": "user", "content": f""" 
                    Convert the query {query} into a Boolean AND/OR format with query expansion. Follow these steps:
                        1. Analyze the query into [X] main concepts, prioritizing named entities or core biomedical terms (e.g., drugs, diseases, proteins) over descriptive aspects (e.g., mechanism, effect).
                        2. Expand each concept with up to 3 synonyms (including the original term):
                            If no suitable synonyms are found, use only the original term.
                            If only 1-2 suitable synonyms are found, include them.
                        3. Combine synonyms within the same concept using OR.
                        4. Combine the concepts using AND.
                    """}
                ],
                response_format=COT,
            )
            response = completion.choices[0].message.parsed

            return response
        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)


def openai_classification(question: str, model_name: str = 'gpt-4o-mini', retries: int =5, temperature: float = 0.0) -> str:
    
    msg = [
        {"role": "system", "content": """You are an expert in biomedical question classification.

        Given a biomedical question, determine whether answering it requires single-hop or multi-hop reasoning.

        Classification criteria:
        - Single-hop: The question involves one entity, or multiple entities that can all be resolved using information from a single document.
        - Multi-hop: The question involves two or more entities and requires combining information retrieved from multiple documents to answer."""},
        {"role": "user", "content": f" Classify the question '{question}' as either 'single' or 'multi' "}
    ]
    max_token = get_output_token_openai(msg, 16384, 128000)
    response = "Empty"

    for i in range (retries):
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            completion = client.chat.completions.create(
                model=model_name,
                messages=msg,
                max_tokens = max_token,
                temperature = temperature,
                # max_tokens=100,
            )
            response = completion.choices[0].message.content
            # self.ideal_answer = response
            response = response.lower()
            if 'single' in response:
                return 'single'
            elif 'multi' in response:
                return 'multi'
            
            # Default return single
            return 'single'

        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)

    return response

def openai_decomposition(question: str, model_name: str = 'gpt-4o-mini', retries: int =5, temperature: float = 0.0) -> list[str]:
    # client = OpenAI(api_key=QA_KEY)

    class Decomposition(BaseModel):
        single_hop_questions: list[str]
    msg = [
        {"role": "system", "content": "You are assigned a multi-hop question decomposition task. You should decompose the given multi-hop question into multiple single-hop questions, and such that you can answer each single-hop question independently."},
        {"role": "user", "content": f" Decompose the multi-hop question '{question}' into single-hop questions. "}
    ]
    max_token = get_output_token_openai(msg, 16384, 128000)
    response = "Empty"

    for i in range(retries):
        try:
            client = OpenAI(api_key=OPENAI_KEY)
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=msg,
                response_format=Decomposition,
                max_tokens = max_token,
                temperature = temperature,
            )
            response = completion.choices[0].message.parsed

            return response.single_hop_questions
        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)