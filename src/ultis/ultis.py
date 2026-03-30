import tiktoken
# import re
# abbreviations = {"mr.", "mrs.", "dr.", "e.g.", "i.e.", "etc.", "vs.", "u.s.", "a.m.", "p.m."}
from sentence_transformers import SentenceTransformer
import torch

def paragraph_to_sentences(paragraph: str, model, join_by: str | None = '\n') -> list:
    doc = model(paragraph)
    if join_by is not None:
        return join_by.join([sent.text for sent in doc.sents])
    return [sent.text for sent in doc.sents]

def list_para_to_list_sentences(list_para: list, model) -> list:
    result = []
    if len(list_para) == 0:
        return result
    else:
        for para in list_para:
            lst = [paragraph_to_sentences(p, model, join_by='\n') for p in para]
            result.append(lst)
    return result



def get_output_token_openai(messages, 
                                max_context_token: int = 128000, 
                                max_output_token: int = 16384):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error occurred while getting encoding: {e}")
        print("using local encoding")
        encoding = tiktoken.get_encoding("cl100k_base")
        return 0

    input_tokens = sum(len(encoding.encode(m["content"])) for m in messages)

    # Giữ tổng token trong giới hạn 128000
    output_token = min(max_output_token, max_context_token - input_tokens-10)
    return output_token

def get_top_k_snippet(question: str, doc: list[str], model: SentenceTransformer, split_sentence_model, k: int = 20):
    try:
        if len(doc) == 0 or doc is None:
            return []
        sentences = [sent.text for d in doc for sent in split_sentence_model(d).sents]
        
        enc_sentences = model.encode(sentences)
        enc_question = model.encode(question)

        similarities = model.similarity(enc_question, enc_sentences)[0]
        if len(similarities) < k or k == -1:
            k = len(similarities)
        values, idx = torch.topk(similarities, k)
        k_snippet = [sentences[i] for i in idx]
        return k_snippet
    except Exception as e:
        print(f"Error: {e}")
        return doc

def top_k_bi_encoder(question: str, doc: list[str], bi_encoder_model, split_sentence_model, k: int = 20):
    try:
        if len(doc) == 0 or doc is None:
            return []
        sentences = [sent.text for d in doc for sent in split_sentence_model(d).sents]
        
        enc_sentences = bi_encoder_model.encode(sentences, convert_to_tensor=True)
        enc_question = bi_encoder_model.encode(question, convert_to_tensor=True)

        similarities = torch.nn.functional.cosine_similarity(enc_question, enc_sentences)
        if len(similarities) < k or k == -1:
            k = len(similarities)
        values, idx = torch.topk(similarities, k)
        k_snippet = [sentences[i] for i in idx]
        return k_snippet
    except Exception as e:
        print(f"Error get top_k_bi_encoder: {e}")
        return doc

def top_k_cross_encoder(question: str, doc: list[str], cross_encoder_model, bi_encoder_model, split_sentence_model, k_cross: int = 20, k_bi: int = 100):
    try:
        if len(doc) == 0 or doc is None:
            return []
        #top_k_bi_encoder 100 sentences
        top_k = top_k_bi_encoder(question, doc, bi_encoder_model, split_sentence_model, k_bi)
        # debug
        # print(f'top_k_bi_encoder: {top_k}')
        
        # top_k_cross_encoder k sentences
        pairs = [[question, sentence] for sentence in top_k]
        scores = cross_encoder_model.predict(pairs)
        
        if len(scores) < k_cross or k_cross == -1:
            k_cross = len(scores)
        values, idx = torch.topk(torch.tensor(scores), k_cross)
        k_snippet = [top_k[i] for i in idx]
        return k_snippet
    except Exception as e:
        print(f"Error get top_k_cross_encoder: {e}")
        return doc  