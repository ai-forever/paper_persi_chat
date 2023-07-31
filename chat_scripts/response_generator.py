from collections import Counter
import re
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer
)


class ResponseGeneratorDialoGPT:
    def __init__(self, model_name_or_path="microsoft/DialoGPT-medium", device='cuda:0'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        self.device = device
        
        self.special_tokens_list = []
        for el in self.tokenizer.special_tokens_map.values():
            if isinstance(el, list):
                self.special_tokens_list.extend(el)
            else:
                self.special_tokens_list.append(el)
        
    def generate_top(self, text, num_beams=4,  max_source_len=512, max_target_length=512, top_k=50, top_p=1, repetition_penalty=1.):
        inputs = self.tokenizer([text], max_length=max_source_len, return_tensors="pt").to(self.device)
        #print('INP LEN', len(inputs["input_ids"][0]))
        
        input_tensor = inputs["input_ids"]
        input_tensor[0][-1] = self.tokenizer.encode(['<INPUTEND>'])[0]
        
        summary_ids = self.model.generate(input_tensor, do_sample=False, num_beams=num_beams,
                                         max_length=max_target_length, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                         pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        
        pred = self.tokenizer.batch_decode(summary_ids[..., inputs["input_ids"].shape[-1]:],
                                      clean_up_tokenization_spaces=False)[0]
        pred = re.sub(r'\s+', ' ', pred).replace(' <EOS>', '').strip()
        pred = re.split('|'.join(self.special_tokens_list), pred)[0] # is not needed for the custom model
        
        #TODO: log
        print(text)
        print(pred)
        
        return pred
        
    def predict(self, question, span, grounding, history, version='v1'):
        history = ' <UTTERSEP> '.join(history)
        history = re.sub(r'\s+', ' ', history)
        span = re.sub(r'\s+', ' ', span)
        if version == 'v1':
            gpt_input = ' <SEP> '.join([question, span, grounding, history]) + ' <INPUTEND> '
        elif version == 'v2':
            gpt_input = ' <SEP> '.join([question, span, '']) + ' <INPUTEND> ' # history
        else:
            raise Exception('Not known version')
        pred = self.generate_top(gpt_input, top_k=50, num_beams=10, repetition_penalty=1.)
        repetition_penalty = 1.
        while (len(pred.split()) == 0 or Counter(pred.split()).most_common(1)[0][1] > 7) and repetition_penalty < 10.:
            repetition_penalty += 2.
            pred = self.generate_top(gpt_input, top_k=50, num_beams=10, repetition_penalty=repetition_penalty)
        return pred
    
    
class ResponseGeneratorBart:
    def __init__(self, model_name_or_path="facebook/bart-large", device='cuda:0'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        self.device = device
        
    def generate_top(self, text, max_source_len=512, max_target_length=512, num_beams=10):
        inputs = self.tokenizer([text], max_length=max_source_len, return_tensors="pt").to(self.device)
        
        input_tensor = inputs["input_ids"]
        
        summary_ids = self.model.generate(
            input_tensor,
            num_beams=num_beams,
            do_sample=False,
            max_length=max_target_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        pred = self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        #TODO: log
        print(text)
        print(pred)
        
        return pred
        
    def predict(self, question, span, grounding, history, version='v1'):
        history = ' <UTTERSEP> '.join(history)
        history = re.sub(r'\s+', ' ', history)
        span = re.sub(r'\s+', ' ', span)
        if version == 'v1':
            gpt_input = ' <SEP> '.join([question, span, grounding, history])
        elif version == 'v2':
            gpt_input = ' <SEP> '.join([question, span, '']) #history
        else:
            raise Exception('Not known version')
        pred = self.generate_top(gpt_input, num_beams=10)
        return pred
