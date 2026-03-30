from rouge_metric import PyRouge

from ultis.ultis import list_para_to_list_sentences, paragraph_to_sentences
import spacy

def rouge2_su4(hypotheses: list, references: list[list], return_pr=False, split_sentence_model: str = None):
    rouge = PyRouge(rouge_n=(2),
                rouge_su=True, skip_gap=4)
    
    if len(hypotheses) != len(references):
        raise ValueError('The number of hypotheses and references should be the same')
    
    if split_sentence_model is None:
        name = "en_core_sci_md"
        print(f'Load spacy model {name}')
        split_sentence_model = spacy.load(name)

    hypotheses = [paragraph_to_sentences(hypo, split_sentence_model, '\n') for hypo in hypotheses]
    references = list_para_to_list_sentences(references, split_sentence_model)
    
    # print(hypotheses)
    # print(references)
    # calculate rouge-1, rouge-2, rouge-su4 scores
    scores = rouge.evaluate(hypotheses, references)
    
    if return_pr:
        return scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-su4']['p'], scores['rouge-su4']['r']
    return scores['rouge-2']['r'], scores['rouge-2']['f'], scores['rouge-su4']['r'], scores['rouge-su4']['f']