import os
from dotenv import load_dotenv
from Bio import Entrez
import time
from ultis.openai import openai_query_expansion
from ultis.gemini_api import gemini_query_expansion

load_dotenv()
# Set the email address to avoid any potential issues with Entrez
Entrez.email = os.getenv('ENTREZ_EMAIL')
Entrez.api_key = os.getenv('ENTREZ_API_KEY')

def get_abstracts(id_list: list, retries: int = 5, max_doc: int = 25) -> list:
    abstracts = []
    ls = id_list.copy()
    for i in ls:
        if "pubmed" in i:
            i = i.split('/')[-1]

    if len(id_list) > max_doc and max_doc != -1:
        id_list = id_list[:max_doc]
    for idx, pmid in enumerate(id_list):
        for i in range(retries):
            try:
                handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml')
                # time.sleep(1)
                records = Entrez.read(handle)
            
                # Process each PubMed article in the response
                for record in records['PubmedArticle']:
                    abstract = ' '.join(record['MedlineCitation']['Article']['Abstract']['AbstractText']) if 'Abstract' in record['MedlineCitation']['Article'] and 'AbstractText' in record['MedlineCitation']['Article']['Abstract'] else ''
                    abstracts.append(abstract)
                break

            except Exception as e:
                print(f"Attempt {i+1}: Error - {e}")
                time.sleep(1+i)  # Exponential backoff (1, 2, 3, ... giây)

    return abstracts

def retrieve_pubmed(question: str, max_ret: int = 25, retries: int = 5) -> list:
    id_list = []
    for i in range(retries):
        try:
            handle = Entrez.esearch(db='pubmed', retmax=max_ret, maxdate="2024/12/31", term=question, sort="relevance")
            record = Entrez.read(handle)
            id_list = record['IdList']
            break
        except Exception as e:
            print(f"Attempt {i+1}: Error - {e}")
            time.sleep(1+i)
    if id_list == [] or id_list is None:
        return []
    return id_list

def Retrival_Pubmed(question: str, max_ret: int = 11, qe: bool = False, retries: int = 5, synonym_number: int = 5) -> list:
    q = question
    if qe:
        q = openai_query_expansion(question)
        
        synonyms = q.concepts
        terms = []
        for s in synonyms:
            term = s.expanded_terms
            t_list = term.split('OR')
            new_term = t_list[:synonym_number] if len(t_list) > synonym_number else t_list
            new_term = ' OR '.join(new_term)
            terms.append('(' + new_term + ')')
        q = ' AND '.join(terms)
    id_list = []
    id_list = retrieve_pubmed(question=q, max_ret=max_ret, retries=retries)
    
    while id_list == [] or id_list is None:
        if len(q) == 0:
            print('No document found')
            print(question)
            return [], q
        
        print('No document found, try to reduce condition')
        q = 'AND'.join(q.split('AND')[:-1])
        id_list = retrieve_pubmed(question=q, max_ret=max_ret, retries=retries)
    
    return id_list, q

def Retrival_Pubmed_gemini(question: str, max_ret: int = 11, qe: bool = False, retries: int = 5, synonym_number: int = 5) -> list:
    q = question
    if qe:
        q = gemini_query_expansion(question)

        synonyms = q.concepts
        terms = []
        for s in synonyms:
            term = s.expanded_terms
            t_list = term.split('OR')
            new_term = t_list[:synonym_number] if len(t_list) > synonym_number else t_list
            new_term = ' OR '.join(new_term)
            terms.append('(' + new_term + ')')
        q = ' AND '.join(terms)

    id_list = []
    id_list = retrieve_pubmed(question=q, max_ret=max_ret, retries=retries)
    
    while id_list == [] or id_list is None:
        if len(q) == 0:
            print('No document found')
            print(question)
            return [], q
        
        print('No document found, try to reduce condition')
        q = ' AND '.join(q.split(' AND ')[:-1])
        id_list = retrieve_pubmed(question=q, max_ret=max_ret, retries=retries)
    
    return id_list, q

