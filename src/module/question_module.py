from retrieval import retrieval
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

# import torch
from ultis.openai import openai_qa_other, openai_qa_summary, openai_classification, openai_decomposition
from ultis.ultis import get_top_k_snippet, get_output_token_openai, top_k_cross_encoder
from ultis.gemini_api import *
import json
from tqdm import tqdm
import os



class ListQuestion():
    def __init__(self, 
                 list_questions: list, 
                 document_type: str = 'abstract', 
                 max_ret: int = 25, 
                 phase : str = 'A+', 
                 model_name: str = 'gpt-4o-mini',
                 temperature: float = 0.0,
                 split_sentence_model: str = "en_core_sci_md",
                 sentence_transformer_model: str = 'all-MiniLM-L12-v2',
                 top_k_snippet: int = 20,
                 qe: bool = False, 
                 next_query: bool = False,
                 qe_sub: bool = False,
                 sub_ret: int = 10,
                 filename: str = None,
                 default: bool = False,
                 synonym_number: int = 5,
                 ): 
        print('Creating questions...')
        self.default = default
        self.split_sentence_model = spacy.load(split_sentence_model)
        self.sentence_transformer_model = SentenceTransformer(sentence_transformer_model) if sentence_transformer_model != 'None' else None

        self.questions = []
        for q in tqdm(list_questions):
            tpm = Question(q, 
                           document_type=document_type, 
                           max_ret=max_ret, 
                        #    max_doc = max_doc, 
                           phase = phase,
                           qe = qe,
                           top_k_snippet = top_k_snippet,
                           sentence_transformer_model = self.sentence_transformer_model,
                           split_sentence_model = self.split_sentence_model,
                           qe_sub=qe_sub,
                           sub_ret=sub_ret,
                           synonym_number=synonym_number,
                           )
            self.questions.append(tpm)

        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.json_path = 'output/' + f'_{phase}' + f'_max_ret_{max_ret}' + f'_{document_type}'
        self.model_name = model_name
        self.temperature = temperature
        self.next_query = next_query
        # self.qe_sub = qe_sub
        if top_k_snippet != -1:
            # self.get_top_k_snippet(self.sentence_transformer_model, k = top_k_snippet)
            self.json_path += f'_top_{top_k_snippet}_snippet'
        if next_query:
            self.json_path += '_next_query'
        if qe:
            self.json_path += '_qe'
        if qe_sub:
            self.json_path += '_qe_sub'
        self.json_path += '.json'
        if filename is not None:
            if '.json' not in filename:
                filename += '.json'
            self.json_path = filename
    
    def openai_qa(self, retries: int =5):
        print('OpenAI QA...')
        for q in tqdm(self.questions):
            if self.default:
                q.openai_qa_default(model_name = self.model_name,
                                       retries = retries,
                                       temperature = self.temperature,)
            elif self.next_query:
                q.openai_qa_next_query(model_name = self.model_name, 
                                       retries = retries, 
                                       temperature = self.temperature,
                                    #    qe_sub = self.qe_sub,
                                       )
            else:
                q.openai_qa(model_name = self.model_name, 
                        temperature = self.temperature, 
                        retries = retries,
                        # next_query = self.next_query,
                        )
    
    def openai_mibi_qa(self, retries: int =5):
        print('OpenAI MIBI QA...')
        for q in tqdm(self.questions):
            if self.default:
                q.openai_qa_default(model_name = self.model_name,
                                       retries = retries,
                                       temperature = self.temperature,)
            elif self.next_query:
                q.openai_qa_next_query(model_name = self.model_name, 
                                       retries = retries, 
                                       temperature = self.temperature,
                                    #    qe_sub = self.qe_sub,
                                       )
            else:
                q.openai_qa(model_name = self.model_name, 
                        temperature = self.temperature, 
                        retries = retries,
                        # next_query = self.next_query,
                        )
            
    def gemini_qa(self, retries: int =5):
        print('Gemini QA...')
        for q in tqdm(self.questions):
            if self.next_query:
                q.gemini_qa_next_query(retries = retries)


    
    def __json__(self):
        ls = [q.__json__() for q in self.questions]
        return {'questions': ls}
    
    def __json2__(self):
        ls = [q.__json2__() for q in self.questions]
        return {'questions': ls} 

    def save_json(self, path: str = None):
        if path is None:
            path = self.json_path
        with open("all_info.json", 'w', encoding='utf-8') as file:
            json.dump(self.__json2__(), file, indent=4)

        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.__json__(), file, indent=4)
        

class Question():
    def __init__(self, 
                 data: dict, 
                 max_ret: int = 25, 
                #  max_doc: int = 10, 
                 document_type: str = 'abstract', 
                 sentence_transformer_model: SentenceTransformer | str = 'None',
                 phase: str = 'A+',
                 split_sentence_model = None,
                 qe: bool = False,
                 top_k_snippet: int = 20,
                 qe_sub: bool = False,
                 sub_ret: int = 10,
                 synonym_number: int = 5,
                 ):
        
        self.question = data['body']
        self.type = data['type']
        self.id = data['id']
        self.phase = phase
        # self.concept = data['concepts']
        self.documents = []
        self.snippet = []
        # self.sentence_transformer_model = sentence_transformer_model
        self.ideal_answer = ""
        self.exact_answer = ""
        self.synonym_number = synonym_number

        self.sentence_transformer_model = sentence_transformer_model
        self.split_sentence_model = split_sentence_model
        self.max_ret = max_ret
        self.document_type = document_type
        self.top_k_snippet = top_k_snippet
        self.qe = qe
        self.qe_sub = qe_sub
        self.sub_ret = sub_ret

        self.final_msg = ""
        self.ret_doc = []
        self.top_snippet = []
        self.retrieve_queries = []
        self.question_classification = ""
        self.sub_question = []
        self.synnonyms = []

        
        if phase == 'B':
            if 'ideal_answer' in data.keys():
                self.golden_ideal_answer = data['ideal_answer']
            else:
                self.golden_ideal_answer = ""
            if 'exact_answer' in data.keys():
                self.golden_exact_answer = data['exact_answer'] if self.type in ['factoid', 'list', 'yesno'] else ""
            else:
                self.golden_exact_answer = ""
            self.snippet = [sp['text'] for sp in data['snippets']] if 'snippets' in data.keys() else []
            self.documents = data['documents'] if 'documents' in data.keys() else []

        elif phase == 'A+':
            if 'ideal_answer' in data.keys():
                self.golden_ideal_answer = data['ideal_answer']
            else:
                self.golden_ideal_answer = ""
            if 'exact_answer' in data.keys():
                self.golden_exact_answer = data['exact_answer'] if self.type in ['factoid', 'list', 'yesno'] else ""
            else:
                self.golden_exact_answer = ""

        
    def create_prompt(self, question: str = None, doc: list = None) -> dict:
        if doc is None:
            doc = '\n\n'.join(self.documents)

        if question is None:
            question = self.question
        prompt = f"""Answer the following question in biomedical domain based on the given documents. After get the answer, please provide the exact answer order by the confidence level of entities:
        
                {doc}
                
                QUESTION: {question}
                
                ANSWER:

                EXACT ANSWER:
                """
        if self.type == 'yesno':
            prompt = f"""Answer the following question in biomedical domain based on the given documents. After get the answer, please provide the exact answer only "yes" or "no":
        
                {doc}
                
                QUESTION: {question}
                
                ANSWER:

                EXACT ANSWER:
                """
        elif self.type == 'summary':
            prompt = f"""Answer the following question in biomedical domain based on the given documents:
        
                {doc}
                
                QUESTION: {question}
                
                ANSWER:"""
            
        result = {
            'role': 'user',
            'content': prompt,
        }
        return result

    def create_system_msg(self) -> dict:
        content = 'You are an expert that helps answer questions only in biomedical research and molecular biology. Your task is to answer questions using only the information retrieved from the provided documents.\n'
        if isinstance(content, tuple):
            content = content[0]
        if self.type == 'yesno':
                content += """Provide a clear "Yes" or "No" response, followed by a concise explanation based on the retrieved evidence.
Examples:
QUESION 1: "Can capivasertib be used for breast cancer?"
IDEAL ANSWER 1: "Yes. Capivasertib is effective and be used for treatment of breast cancer"
EXACT ANSWER 1: "yes"

QUESTION 2: "Is there a specific cure for Ehlers-Danlos Syndrome?"
IDEAL ANSWER 2: "No. Currently, there is no specific cure for Ehlers-Danlos Syndrome. Management is focused on treating the various symptoms and complications that can arise from the condition."
EXACT ANSWER 2: "no"

Now, answer the following question in a similar format
"""
        elif self.type == 'factoid':
            content += """You need to answer factoid questions, which require a specific entity (e.g., a disease, drug, or gene), a number, or a short factual expression as an answer
Examples:
QUESION 1: "What is the meaning of the acronym \"TAILS\" used in protein N-terminomics?"
IDEAL ANSWER 1: "TAILS stands for \"Terminal Amine Isotopic Labeling of Substrates\""
EXACT ANSWER 1: "TAILS: Terminal Amine Isotopic Labeling of Substrates"

QUESTION 2: "What is the methyl donor of DNA (cytosine-5)-methyltransferases?"
IDEAL ANSWER 2: "S-adenosyl-L-methionine (AdoMet, SAM) is the methyl donor of DNA (cytosine-5)-methyltransferases. DNA (cytosine-5)-methyltransferases catalyze the transfer of a methyl group from S-adenosyl-L-methionine to the C-5 position of cytosine residues in DNA."
EXACT ANSWER 2: "S-adenosyl-L-methionine"

Now, answer the following question in a similar format
"""
        elif self.type == 'list':
            content += """You need to answer list questions, which require a list of entity names (e.g., genes, proteins, inhibitors) or short factual expressions. The answers should be concise but informative, listing key items and including brief explanations when necessary.
Examples:
QUESION 1: "List signaling molecules (ligands) that interact with the receptor EGFR?"
IDEAL ANSWER 1: "The 7 known EGFR ligands  are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-\u03b1 [TGF-\u03b1], amphiregulin (AREG) and epigen (EPG)."
EXACT ANSWER 1: ["epidermal growth factor", "betacellulin", "epiregulin", "heparin-binding epidermal growth factor", "transforming growth factor-\u03b1", "amphiregulin", "epigen"]

QUESTION 2: "What are the effects of depleting protein  km23-1 (DYNLRB1)  in a cell?"
IDEAL ANSWER 2: "The knockdown of km23-1 results in numerous effects at the cellular level, such as decreased cell migration. Additionaly, km23-1 is involved in signalling pathways and its knockdown results in decreased RhoA activation, inhibition of TGF\u03b2-mediated activation of ERK and JNK, phosphorylation of c-Jun, transactivation of the c-Jun promoter and decreased TGFbeta responses."
EXACT ANSWER 2: ["inhibition of cell migration of human colon carcinoma cells (HCCCs) in wound-healing assays", "decreased RhoA activation", "inhibition of TGF\u03b2-mediated activation of ERK and JNK", "phosphorylation of c-Jun", "transactivation of the c-Jun promoter", "decreased key TGFbeta responses"]

Now, answer the following question in a similar format"""
        elif self.type == 'summary':
            content += 'For summary question, provide a concise yet informative summary that covers the key aspects of the topic. Ensure clarity and coherence while keeping the response brief.'

        # if self.type != 'summary':
        #     content += '\n'
        return {
            'role': 'system',
            'content': content,}

    def create_msg(self, final_query: str = None, doc: list = None) -> list[dict]:
        prompt = self.create_prompt(final_query, doc)

        system_msg = self.create_system_msg()
        msg = [system_msg, prompt]
        max_token = get_output_token_openai(msg, 16384, 128000)
        new_doc = doc
        
        while max_token < 1000:
            new_doc = new_doc[:-1]
            prompt = self.create_prompt(final_query, new_doc)
            msg = [system_msg, prompt]
            max_token = get_output_token_openai(msg, 16384, 128000)
        return [system_msg, prompt]
    
    def retrieve_documents_gemini(self, question: str = None) -> list:
        if question is None:
            question = self.question

        id_list, ret_q = retrieval.Retrival_Pubmed_gemini(question, max_ret = self.max_ret, qe  = self.qe, synonym_number=self.synonym_number)
        doc = retrieval.get_abstracts(id_list, max_doc = self.max_ret)

        if self.document_type == 'snippet':
            doc = get_top_k_snippet(question=question, 
                                    doc=doc, 
                                    model = self.sentence_transformer_model, 
                                    split_sentence_model=self.split_sentence_model, 
                                    k = self.top_k_snippet,
                                    )
        self.top_snippet = doc
        self.retrieve_queries.append({question: ret_q})
        return doc, id_list
        
    def get_documents_gemini(self, question: str = None) -> list:
        if question is None:
            question = self.question

        if self.document_type == 'abstract':
            doc = retrieval.get_abstracts(self.documents, max_doc = self.max_ret,)
            id_list = self.documents
        elif self.document_type == 'snippet':
            doc = self.snippet
            id_list = []
            if len(doc) < 1:
                id_list = retrieval.Retrival_Pubmed_gemini(question, max_ret = self.max_ret, qe  = self.qe, synonym_number=self.synonym_number)
                doc = retrieval.get_abstracts(id_list, max_doc = self.max_ret)

                doc = get_top_k_snippet(question=question, 
                                        doc=doc, 
                                        model = self.sentence_transformer_model, 
                                        split_sentence_model=self.split_sentence_model, 
                                        k = self.top_k_snippet,
                                        )
            elif len(doc) > self.top_k_snippet and self.top_k_snippet!=-1:
                doc = doc[:self.top_k_snippet]

            elif len(doc) >= 1 and len(doc) < self.top_k_snippet and self.top_k_snippet != -1:
                extra_id = retrieval.Retrival_Pubmed_gemini(question, max_ret = self.max_ret, qe  = self.qe, synonym_number=self.synonym_number)
                extra_doc = retrieval.get_abstracts(extra_id, max_doc = self.max_ret)
                extra_doc = get_top_k_snippet(question=question, doc=extra_doc, 
                                                model = self.sentence_transformer_model, 
                                                split_sentence_model=self.split_sentence_model, 
                                                k = self.top_k_snippet-len(doc),
                                                )
                doc += extra_doc
                id_list += extra_id
                # doc = doc[:self.max_ret]
        self.top_snippet = doc
        return doc, id_list

    def retrieve_documents(self, question: str = None) -> list:
        if question is None:
            question = self.question

        id_list, ret_q = retrieval.Retrival_Pubmed(question, max_ret = self.max_ret, qe  = self.qe, synonym_number=self.synonym_number)
        doc = retrieval.get_abstracts(id_list, max_doc = self.max_ret)

        if self.document_type == 'snippet':
            doc = get_top_k_snippet(question=question, 
                                    doc=doc, 
                                    model = self.sentence_transformer_model, 
                                    split_sentence_model=self.split_sentence_model, 
                                    k = self.top_k_snippet,
                                    )
        elif self.document_type == 'bi-cross':
            doc = top_k_cross_encoder(question=question, 
                                    doc=doc,
                                    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2"),
                                    bi_encoder_model = self.sentence_transformer_model,
                                    split_sentence_model=self.split_sentence_model,
                                    k_bi = 100,
                                    k_cross = self.top_k_snippet,)
        self.top_snippet = doc
        self.retrieve_queries.append({question: ret_q})
        return doc, id_list
    
        
    def get_documents(self, question: str = None) -> list:
        if question is None:
            question = self.question

        if self.document_type == 'abstract':
            doc = retrieval.get_abstracts(self.documents, max_doc = self.max_ret,)
            id_list = self.documents
        elif self.document_type == 'snippet':
            doc = self.snippet
            id_list = []
            if len(doc) < 1:
                id_list = retrieval.Retrival_Pubmed(question, max_ret = self.max_ret, qe  = self.qe, synonym_number=self.synonym_number)
                doc = retrieval.get_abstracts(id_list, max_doc = self.max_ret)

                doc = get_top_k_snippet(question=question, 
                                        doc=doc, 
                                        model = self.sentence_transformer_model, 
                                        split_sentence_model=self.split_sentence_model, 
                                        k = self.top_k_snippet,
                                        )
            elif len(doc) > self.top_k_snippet and self.top_k_snippet!=-1:
                doc = doc[:self.top_k_snippet]

            elif len(doc) >= 1 and len(doc) < self.top_k_snippet and self.top_k_snippet != -1:
                extra_id = retrieval.Retrival_Pubmed(question, max_ret = self.max_ret, qe  = self.qe, synonym_number=self.synonym_number)
                extra_doc = retrieval.get_abstracts(extra_id, max_doc = self.max_ret)
                extra_doc = get_top_k_snippet(question=question, doc=extra_doc, 
                                                model = self.sentence_transformer_model, 
                                                split_sentence_model=self.split_sentence_model, 
                                                k = self.top_k_snippet-len(doc),
                                                )
                doc += extra_doc
                id_list += extra_id
                # doc = doc[:self.max_ret]
        self.top_snippet = doc
        return doc, id_list

        # self.documents = '\n\n'.join(self.documents)
    
    def sub_question_qa(self, question: str):
        id_list, ret_q = retrieval.Retrival_Pubmed(question, max_ret = self.sub_ret, qe  = self.qe_sub, synonym_number=self.synonym_number)
        doc = retrieval.get_abstracts(id_list, max_doc = self.sub_ret)
        doc = get_top_k_snippet(question=question, 
                                doc=doc, 
                                model = self.sentence_transformer_model, 
                                split_sentence_model=self.split_sentence_model, 
                                k = self.top_k_snippet,
                                )
        
        msg = [
            {"role": "system", "content": "You are an assistant that helps answer questions in biomedical domain."},
            {"role": "user", "content": f"""Answer the following question in biomedical domain based on the given documents:
        
                {doc}
                
                QUESTION: {question}
                
                ANSWER:"""}
        ]
        ans = openai_qa_summary(msg = msg)
        return ans, doc, ret_q
    
    def sub_question_qa_gemini(self, question: str):
        id_list, ret_q = retrieval.Retrival_Pubmed_gemini(question, max_ret = self.sub_ret, qe  = self.qe_sub, synonym_number=self.synonym_number)
        doc = retrieval.get_abstracts(id_list, max_doc = self.sub_ret)
        doc = get_top_k_snippet(question=question, 
                                doc=doc, 
                                model = self.sentence_transformer_model, 
                                split_sentence_model=self.split_sentence_model, 
                                k = self.top_k_snippet,
                                )
        
        msg = [
            {"role": "system", "content": "You are an assistant that helps answer questions in biomedical domain."},
            {"role": "user", "content": f"""Answer the following question in biomedical domain based on the given documents:
        
                {doc}
                
                QUESTION: {question}
                
                ANSWER:"""}
        ]
        ans = gemini_qa(msg = msg)
        return ans, doc, ret_q

    def get_final_query(self, questions: list, model_name: str = 'gpt-4o-mini'):
        sub_ans = []
        sub_querys = []
        for q in questions:
            if model_name.startswith('gemini'):
                tmp, docs, ret_q = self.sub_question_qa_gemini(q)
            elif model_name.startswith('gpt'):
                tmp, docs, ret_q = self.sub_question_qa(q)
            sub_ans.append(tmp)
            sub_querys.append({q: ret_q})
        self.retrieve_queries.append(sub_querys)

        msg = [
            {"role": "system", "content": f"""You are assigned a multi-hop question refactoring task.
 Given a complex question along with a set of related known information, you are required to refactor the question by
 applying the principle of retraining difference and removing redundancies. Specifically, you should eliminate the content
 that is duplicated between the question and the support information, leaving only the parts of the question that have
 not been answered, and the new knowledge points in the known information. The ultimate goal is to reorganize these
 retrained parts to form a new question.
 You can only generate the question by picking words from the question and known information. You should first pick
 up words from the question, and then from each known info, and concatenate them finally. """},

            {"role": "user", "content": f"""
             Multi-hop question '{self.question}' 
             Support information: {sub_ans}
             """},
        ]
        self.sub_question.append(sub_querys)
        ans = openai_qa_summary(msg = msg)
        return ans

    def openai_qa_default(self, model_name: str = 'gpt-4o-mini', retries: int = 5, temperature: float = 0.0):
        system_msg = self.create_system_msg()
        question = self.question
        prompt = f"""Answer the following question in biomedical domain. After get the answer, please provide the exact answer order by the confidence level of entities:
                QUESTION: {question}
                
                ANSWER:

                EXACT ANSWER:
                """
        if self.type == 'yesno':
            prompt = f"""Answer the following question in biomedical domain. After get the answer, please provide the exact answer only "yes" or "no":
                QUESTION: {question}
                ANSWER:
                EXACT ANSWER:
                """
        elif self.type == 'summary':
            prompt = f"""Answer the following question in biomedical domain:
                QUESTION: {question}
                ANSWER:"""
        p = {
            'role': 'user',
            'content': prompt,
        }

        msg = [system_msg, p]
        # max_token = get_output_token_openai(msg, 16384, 128000)
        # response = ""
        
        if self.type == 'summary':
            ideal_ans = openai_qa_summary(msg = msg, 
                                          model_name = model_name, 
                                          retries = retries, 
                                          temperature = temperature,)
            self.ideal_answer = ideal_ans
        elif self.type == 'yesno':
            ideal_ans, exact_ans = openai_qa_other(msg = msg, 
                                                   model_name = model_name, 
                                                   retries = retries, 
                                                   temperature = temperature,)
            self.ideal_answer = ideal_ans
            self.exact_answer = exact_ans
        
        elif self.type in ['factoid', 'list']:
            ideal_ans, exact_ans = openai_qa_other(msg = msg, 
                                                   model_name = model_name, 
                                                   retries = retries, 
                                                   temperature = temperature,)
            self.ideal_answer = ideal_ans
            self.exact_answer = [[i] for i in exact_ans]

        self.final_msg = msg
            
    def openai_qa_next_query(self, model_name: str = 'gpt-4o-mini', retries: int = 5, temperature: float = 0.0):

        q_type = openai_classification(self.question, model_name = model_name, retries = retries, temperature = 0.0)
        
        question = self.question
        self.question_classification = q_type
        if q_type == 'multi':
            sub_question = openai_decomposition(self.question, model_name = model_name, retries = retries, temperature = 0.0)
            question = self.get_final_query(sub_question)
            
            # self.ideal_answer = final_query

        if self.phase == 'B':
            doc, id_list = self.get_documents()
        else:
            doc, id_list = self.retrieve_documents()
        self.ret_doc = id_list
        
        msg = self.create_msg(question, doc)
        self.final_msg = msg
        if self.type == 'summary':
            ideal_ans = openai_qa_summary(msg = msg, 
                                          model_name = model_name, 
                                          retries = retries, 
                                          temperature = temperature,)
            self.ideal_answer = ideal_ans

        elif self.type == 'yesno':
            ideal_ans, exact_ans = openai_qa_other(msg = msg, 
                                                   model_name = model_name, 
                                                   retries = retries, 
                                                   temperature = temperature,)
            self.ideal_answer = ideal_ans
            self.exact_answer = exact_ans
            
        elif self.type == 'factoid' or self.type == 'list':
            ideal_ans, exact_ans = openai_qa_other(msg = msg, 
                                                   model_name = model_name, 
                                                   retries = retries, 
                                                   temperature = temperature,)
            self.ideal_answer = ideal_ans
            self.exact_answer = [[i] for i in exact_ans]

    def openai_qa(self, model_name: str = 'gpt-4o-mini', retries: int = 5, temperature: float = 0.0,):
        if self.phase == 'B':
            doc, id_list = self.get_documents()
        else:
            doc, id_list = self.retrieve_documents()
        self.ret_doc = id_list
        if self.type == 'summary':
            msg = self.create_msg(doc = doc)
            ideal_ans = openai_qa_summary(msg = msg, 
                                          model_name = model_name, 
                                          retries = retries, 
                                          temperature = temperature,)
            self.ideal_answer = ideal_ans
        elif self.type == 'yesno':
            msg = self.create_msg(doc = doc)
            ideal_ans, exact_ans = openai_qa_other(msg = msg, 
                                                   model_name = model_name, 
                                                   retries = retries, 
                                                   temperature = temperature,)
            self.ideal_answer = ideal_ans
            self.exact_answer = exact_ans
        
        elif self.type in ['factoid', 'list']:
            msg = self.create_msg(doc = doc)
            ideal_ans, exact_ans = openai_qa_other(msg = msg, 
                                                   model_name = model_name, 
                                                   retries = retries, 
                                                   temperature = temperature,)
            self.ideal_answer = ideal_ans
            self.exact_answer = [[i] for i in exact_ans]

        self.final_msg = msg
    
    def print_info(self, all = False):
        if all:
            print(f"Question: {self.question}")
            print(f"Type: {self.type}")
            print(f"Golden Answer: {self.golden_ideal_answer}")
            print(f"Snippets: {self.snippet}")
            print(f"Ideal Answer: {self.ideal_answer}")
            print(f"Exact Answer: {self.exact_answer}")
        else:
            print(f"Question: {self.question}")
            print(f"Ideal Answer: {self.ideal_answer}")
            print(f"Exact Answer: {self.exact_answer}")

    def gemini_qa_next_query(self,  retries: int = 5):
        q_type = gemini_classification(self.question, retries = retries)
        self.question_classification = q_type
        question = self.question
        if q_type == 'multi':
            # print('Multi-hop question detected.')
            sub_question = gemini_decomposition(self.question, retries = retries)
            question = self.get_final_query(sub_question)
        # else:
        #     print('Single-hop question detected.')
            # self.ideal_answer = final_query

        if self.phase == 'B':
            doc, id_list = self.get_documents_gemini()
        else:
            doc, id_list = self.retrieve_documents_gemini()
        self.ret_doc = id_list
        
        msg = self.create_msg(question, doc)
        self.final_msg = str(msg)

        msg = str(msg)
        if self.type == 'summary':
            ideal_ans = gemini_qa(msg = msg, retries= retries)
            self.ideal_answer = ideal_ans

        elif self.type == 'yesno':
            ideal_ans, exact_ans = gemini_qa_ideal_exact_answer(msg = msg, retries = retries)
            self.ideal_answer = ideal_ans
            self.exact_answer = exact_ans
            
        elif self.type == 'factoid' or self.type == 'list':
            ideal_ans, exact_ans = gemini_qa_ideal_exact_answer(msg = msg, retries = retries)

            self.ideal_answer = ideal_ans
            self.exact_answer = [[i] for i in exact_ans]
    
    def __json2__(self):
        ret_doc = self.ret_doc
        for i in range(len(ret_doc)):
            if "pubmed" not in ret_doc[i]:
                ret_doc[i] = "https://pubmed.ncbi.nlm.nih.gov/" + ret_doc[i]
        ans = {
            'type': self.type,
            'body': self.question,
            'id': self.id,
            'ideal_answer': self.ideal_answer,
            'exact_answer': "",
            'documents': self.documents,
            'snippets': self.snippet,
            'top_snippet': self.top_snippet,
            'ret_doc': ret_doc,
            'final_msg': self.final_msg,
            'top_snippet': self.top_snippet,
            'retrieve_queries': self.retrieve_queries,
            'question_classification': self.question_classification,
            'sub_question': self.sub_question,
            'synnonyms': self.synnonyms,
        }
        if self.type == 'list':
            ans['exact_answer'] = self.exact_answer
        elif self.type == 'factoid':
            ans['exact_answer'] = self.exact_answer
            if len(self.exact_answer) > 5:
                ans['exact_answer'] = self.exact_answer[:5]
                
        elif self.type == 'yesno':
            ans['exact_answer'] = self.exact_answer[0] if len(self.exact_answer) > 0 else ""
        
        return ans
    
    def __json__(self):
        ans = {
                'type': self.type,
                'body': self.question,
                'id': self.id,
                'ideal_answer': self.ideal_answer,
                'exact_answer': [[]],
                'documents': [],
                'snippets': [],
        }

        if self.type == 'list':
            ans['exact_answer'] = self.exact_answer
        elif self.type == 'factoid':
            ans['exact_answer'] = self.exact_answer
            if len(self.exact_answer) > 5:
                ans['exact_answer'] = self.exact_answer[:5]
        elif self.type == 'yesno':
            ans['exact_answer'] = self.exact_answer[0] if len(self.exact_answer) > 0 else ""
        return ans