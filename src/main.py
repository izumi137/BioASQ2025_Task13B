# from retrieval import retrieval
import json
from module.question_module import ListQuestion
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for sampling')
    # parser.add_argument('--use_retrieved', type=bool, default=True, help='Use retrieved data or not')
    parser.add_argument('--split_sentence_model', type=str, default='en_core_sci_md', help='Split sentence model scispacy')
    parser.add_argument('--max_ret', type=int, default=25, help='Max retrieval')
    # parser.add_argument('--max_doc', type=int, default=10, help='Max document per question, -1 to use all document')
    parser.add_argument('--data_path', type =str , help='path to data json file')
    parser.add_argument('--phase', type = str, default = "B", help='phase "A+" or "B"')
    parser.add_argument('--document_type', type = str, default = "abstract", help='document type: "abstract" or "   " or "both"')
    parser.add_argument('--valid_mode', type = str, default = "per_question", help='valid mode: "all" or "per_question"')
    parser.add_argument('--percent_data', type = float, default=1.0, help='first % data to run, range from 0.0 to 1.0')
    parser.add_argument('--top_k_snippet', type = int, default=-1, help='top k snippet to use, -1 to use all')
    parser.add_argument('--next_query', type = bool, default=False, help='Use next-query module or not')
    parser.add_argument('--sentence_transformer_model', type = str, default='None', help='Sentence transformer model')
    parser.add_argument('--qe', type = bool, default=True, help='Use query expansion or not')
    parser.add_argument('--qe_sub', type = bool, default=False, help='Use query expansion for sub-question or not')
    parser.add_argument('--sub_ret', type = int, default=10, help='Number of sub-question retrieved document')
    parser.add_argument('--output', type = str, default=None, help='Output file name')
#     parser.add_argument('--', type = bool, default=False, help='Using snippet or not')
    parser.add_argument('--synonym_number', type = int, default=5, help='Number of synonyms for query expansion')
    parser.add_argument('--submit', type = bool, default = False, help='No validation, only submit')
    parser.add_argument('--default', type = bool, default = False, help='Use default OpenAI QA method, without retrieval')
    parser.add_argument('--model',  type = str, default='openai', help='Which LLM to use: openai/ gemini/ openai-mibi')
    # parser.add_argument('--vectordb', type = str, default = , help='Path to submit file')
#     parser.add_argument('', type = , help='')
#     parser.add_argument('', type = , help='')

    

#     parser.add_argument('', type = , help='')

    args = parser.parse_args()
    
    with open(args.data_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    length = int(len(data['questions'])*args.percent_data)
    
    
    q = ListQuestion(
        data['questions'][-length:], 
        # data['questions'][20:30], 
        document_type=args.document_type, 
        max_ret=args.max_ret,
        # max_doc=args.max_doc,
        split_sentence_model=args.split_sentence_model,
        model_name=args.model_name,
        temperature=args.temperature,
        sentence_transformer_model=args.sentence_transformer_model,
        top_k_snippet=args.top_k_snippet,
        phase = args.phase,
        next_query = args.next_query,
        qe = args.qe,
        qe_sub = args.qe_sub,
        sub_ret = args.sub_ret,
        filename = args.output,
        default = args.default,
        synonym_number = args.synonym_number,
    )

    if args.model.lower() == 'gemini':
        q.gemini_qa()
    elif args.model.lower() == 'openai-mibi':
        q.openai_mibi_qa()
    else:
        q.openai_qa()
    
    q.save_json()
    submit = args.submit
    if submit == False:
        q.valid(valid_mode = args.valid_mode)
    


    
    