import pickle
import torch
from transformers import pipeline
from tqdm import tqdm

from nltk.tokenize.punkt import PunktSentenceTokenizer


class DebertaQA:
    def __init__(self, model_name='deepset/deberta-v3-base-squad2', device='cuda:0'):
        self.model = pipeline("question-answering", model=model_name, device=device)
        self.device = device
        
    def predict(self, question, context, max_tokens=384, min_tokens_diff=64, min_score=1e-5):
        # topk - how much answers to return
        result = self.model(question, context, max_seq_len=max_tokens, doc_stride=min(max_tokens//2, min_tokens_diff), max_answer_len=max_tokens)
        if result['score'] > min_score:
            result['start_pos'] = result.pop('start')
            result['end_pos'] = result.pop('end')
            result['text'] = result.pop('answer')
            return result
        return {'start_pos': 0, 'end_pos': 0, 'text': '', 'score': 0}
    
    def extract_grounding(self, question, context,
                          max_tokens=500, min_tokens_diff=200, return_response_span=False, sents_around=1, min_score=1e-4):
        response_span = self.predict(question, context, max_tokens=max_tokens, min_tokens_diff=min_tokens_diff, min_score=min_score)
        
        sents_spans = list(PunktSentenceTokenizer().span_tokenize(context))
        sents_in = []
        for i, (start, end) in enumerate(sents_spans):
            if (response_span['start_pos'] >= start and response_span['start_pos'] < end) or \
                    (response_span['end_pos'] > start and response_span['end_pos'] <= end):
                sents_in.append(i)
        
        if len(sents_in) > 0 and len(response_span['text']) > 0:
            grounding_start = min(sents_spans[max(min(sents_in) - sents_around, 0)][0], response_span['start_pos'])
            grounding_end = max(sents_spans[min(max(sents_in) + sents_around, len(sents_spans) - 1)][1], response_span['end_pos'])

            result = {'start_pos': grounding_start,
                      'end_pos': grounding_end,
                      'text': context[grounding_start:grounding_end],
                      'score': response_span['score']}
        
        else:
            result = response_span
        
        if return_response_span:
            return result, response_span
        return result