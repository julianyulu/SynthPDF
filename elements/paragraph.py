import numpy as np
from reportlab.platypus import Paragraph
from elements.style_generator import ParagraphStyleGenerator
from elements.utils import random_integer_from_list

class SynthParagraph:
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    cn_char_cache = []
    def __init__(self, config):
        self.config = config
        
    @property
    def paragraph(self):
        seperator = [',', '，', ':', '：', '.', '。', '!', '！', '?', '？', ' ']
        #seperator = [',', '.', '!']
        cfg_para_long = self.config['long']
        cfg_para_short = self.config['short']
        prob_long = cfg_para_long['prob']
        prob_short = cfg_para_short['prob']
        prob_short = prob_short / (prob_short + prob_long)

        # select by prob to have long/short paragraph 
        cfg_select = cfg_para_short if np.random.random() < prob_short else cfg_para_long 
        lb_sentence, ub_sentence = cfg_select['sentence_length']
        n_sentences = random_integer_from_list(cfg_select['n_sentences'])
        all_words = [self._gen_random_sentence([lb_sentence, ub_sentence]) for _ in range(n_sentences)]
        text = ''
        for w in all_words:
            text += w
            text += np.random.choice(seperator)
        paragraph_style = ParagraphStyleGenerator(self.config).style()
        
        return Paragraph(text, paragraph_style)


    @property
    def cnChar(self):
        if not self.cn_char_cache:
            with open(self.CN_CHAR_FILE, 'r') as fid:
                content = fid.readlines()
            self.cn_char_cache = [x.strip() for x in content]
        return self.cn_char_cache
    
    def _gen_random_sentence(self, length = [2, 20]):
        word_len = random_integer_from_list(length)                
        word = ''.join(np.random.choice(self.cnChar, size = word_len, replace = True).tolist())
        return word
