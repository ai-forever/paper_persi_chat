import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class BartGenerator:
    def __init__(self, model_name_or_path, cache_dir='~/.cache/huggingface/', device='cuda'):
        self.device = device

        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model =  BartForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir).to(device)
                 
    def __call__(self, texts, max_in_length=512, max_out_length=128, num_beams=2,
                 temperature=1., repetition_penalty=1., num_return_sequences=1):
        if type(texts) != list:
            texts = [texts]
        inputs = self.tokenizer(texts, max_length=max_in_length, return_tensors="pt", truncation=True).to(self.device)
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=num_beams, max_length=max_out_length,
                                         temperature=temperature, repetition_penalty=repetition_penalty,
                                         num_return_sequences=num_return_sequences)
        pred = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return pred
    