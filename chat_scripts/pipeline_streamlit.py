import streamlit as st

from deberta_qa import DebertaQA
from chat_utils import check_text_coverage, join_segments
from summary_generation_inference import BartGenerator
from response_generator import ResponseGeneratorDialoGPT, ResponseGeneratorBart

import sys
sys.path.insert(0, 'dialogue_discourse_parser')
checkpoint_path = '../checkpoint/'

from agreement_classifier import AgreementClassifier
from dialogue_discourse_parser.discodial_parser import DiscoDialParser

from functools import reduce
import portion as P
import re

try:
    from allennlp_models import pretrained
except:
    print('allennlp is not installed => the coreference resolver cant be used')
    

class PersistentChatBot:
    def __init__(self, solve_corefs=True, device='cuda:0'):
        self.discodial_parser = DiscoDialParser(
            train_file=checkpoint_path + 'convokit_dials_train.json',
            prefix=checkpoint_path + 'convokit_50',
            word_vector=checkpoint_path + 'glove.6B.100d.txt'
        )
        
        self.agreement_classifier = AgreementClassifier(
            model_path=checkpoint_path + 'sentence_bert_disco',
            classifier_path=checkpoint_path + 'classifier_disco.pth',
            device=device
        )
        
        self.deberta_qa = DebertaQA(
            model_name=checkpoint_path + 'deberta_qa/',
            device=device
        )
        
        self.summarizer = BartGenerator(checkpoint_path + "distilbart_summarizer",
                                       device=device)
        
        self.response_generator = ResponseGeneratorBart(
            model_name_or_path=checkpoint_path + 'bart_response_generator',
            device=device
        )
        
        self.solve_corefs = solve_corefs
        if solve_corefs:
            self.coref_predictor = pretrained.load_predictor('coref-spanbert')
             
    def init_paper_dialogue(self, paper, max_negative=2, sents_around_init=0, text_coverage_limit=0.8):
        self.max_negative = max_negative
        self.sents_around_init = sents_around_init
        
        self.text_coverage = [] # for span coverage in grounding
        self.text_coverage_limit = text_coverage_limit
        
        self.dialogue_history = []
        self.count_negative = 0
        self.chat_end = False
        self.segmented_paper = join_segments(paper['segments'])
        self.paper_title = paper['title']
        self.dialogue_state = 'intro'
        self.cur_segment_num = -1
        
        self.paper_meta = paper['meta'] if 'meta' in paper else {}
    
    def chat_intro(self):
        chat_intro = f'''Let's discuss the paper named "{self.paper_title}"'''
        if len(self.segmented_paper) > 0:
            chat_intro += '\n' + self.summarizer(self.segmented_paper[0][0])[0]
        else:
            self.chat_end = True
        self.dialogue_history.append({'speaker': 'bot', 'text': chat_intro})

        self.cur_segment_num = 0
        self.dialogue_state = 'qa'
        return chat_intro
    
    def suggest_block(self):
        block_intro_q = f'Do you want to discuss a paper segment containing the following {"sections" if len(self.segmented_paper[self.cur_segment_num][1]) > 1 else "section"}: {", ".join([repr(el["title"]) for el in self.segmented_paper[self.cur_segment_num][1]])}?'
        self.dialogue_history.append({'speaker': 'bot', 'text': block_intro_q})

        self.dialogue_state = 'block_suggested'
        return block_intro_q
    
    def chat_ending(self, scenario='good'):
        # scenario = [good, negative]
        self.chat_end = True
        if scenario == 'good':
            chat_end =  "We have discussed all sections of the paper. Bye!"
        else:
            chat_end =  "I'm sorry that I caused negative emotions :("
        return chat_end
    
    def generate_response(self, query=''):
        if self.chat_end:
            return self.chat_ending()
        
        if self.dialogue_state == 'intro':
            return self.chat_intro()
        
        if self.dialogue_state == 'block_suggest':
            return self.suggest_block()
        
        if self.dialogue_state == 'block_suggested':
            self.dialogue_history.append({'speaker': 'user', 'text': query})
            
            if len(query.strip()) == 0:
                user_agree = True
            else:
                try:
                    disco_rel = self.discodial_parser.parse(self.dialogue_history[-10:])[-1][-1]
                except:
                    disco_rel = 'other'
                user_agree = True
                if disco_rel != 'agreement' and len(self.dialogue_history) >= 2:
                    pred_agr = self.agreement_classifier.predict([[self.dialogue_history[-2]['text'],
                                                           self.dialogue_history[-1]['text']]])[0]
                    user_agree = pred_agr[1] > pred_agr[0]
                
            if user_agree:
                block_section_intro = self.summarizer(self.segmented_paper[self.cur_segment_num][0])[0]
                self.dialogue_history.append({'speaker': 'bot', 'text': block_section_intro})
                self.dialogue_state = 'qa'
                return block_section_intro
            
            else:
                self.cur_segment_num += 1
                if self.cur_segment_num < len(self.segmented_paper):
                    return self.suggest_block()
                else:
                    return self.chat_ending()
                
        if self.dialogue_state == 'qa':
            context = self.segmented_paper[self.cur_segment_num][0]
            
            self.dialogue_history.append({'speaker': 'user', 'text': query})
            query_from_user = query
            disco_rel = self.discodial_parser.parse(self.dialogue_history[-10:])[-1][-1]
            
            if disco_rel == 'question':
                if self.solve_corefs:
                    history = re.sub(r'\s+', ' ', ' '.join([el['text'] for el in self.dialogue_history[-4:]]))
                    query_coref_solved = self.coref_predictor.coref_resolved(history + '\n' + query)
                    query_new = query_coref_solved.split('\n')[1].strip()
                    if len(query_new) / len(query) < 1.5:
                        query = query_new
                
                used_context = context

                for max_tokens in [384, 256, 512, 135, 70]:
                    grounding, span = self.deberta_qa.extract_grounding(query,
                                                                        context,
                                                                        max_tokens=max_tokens,
                                                                        return_response_span=True,
                                                                        sents_around=self.sents_around_init)
                    if len(span['text'].strip()) > 0:
                        break
                
                prefix = ''
                if len(span['text'].strip()) == 0:
                    # try to answer using other sections
                    for segment_context, segment_meta in self.segmented_paper:
                        grounding, span = self.deberta_qa.extract_grounding(query,
                                                                            segment_context,
                                                                            max_tokens=384,
                                                                            return_response_span=True,
                                                                            sents_around=self.sents_around_init)
                        
                        if len(span['text'].strip()) > 0:
                            prefix = f'I found relevant information in the {"sections" if len(segment_meta) > 1 else "section"}: ' +\
                                ", ".join([repr(el["title"]) for el in segment_meta]) + '. '
                            used_context = segment_context
                            break
                
                if len(span['text'].strip()) == 0:
                    response = "I don't have enough information to answer this question."
                else:
                    history = [el['text'] for el in self.dialogue_history[-3:-1]]
                    response = self.response_generator.predict(query_from_user, span['text'], grounding['text'], history, version=self.version)
                    response = prefix + response
                
                self.dialogue_history.append({'speaker': 'bot', 'text': response})
                return response

            elif disco_rel == 'negativereaction' and (len(query) > 10 or len(self.dialogue_history) > 3):
                self.count_negative += 1
                if self.count_negative >= self.max_negative:
                    return self.chat_ending(scenario='negative')
                else:
                    response =  "I'm sorry. You can ask me another question or we can move futher."
                    self.dialogue_history.append({'speaker': 'bot', 'text': response})
                    return response
                
            else:
                self.cur_segment_num += 1
                if self.cur_segment_num < len(self.segmented_paper):
                    return self.suggest_block()
                else:
                    return self.chat_ending()
    