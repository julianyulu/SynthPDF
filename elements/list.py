import numpy as np
from reportlab.platypus import Paragraph
from elements.paragraph import SynthParagraph
from elements.style_generator import ParagraphStyleGenerator
from elements.utils import random_integer_from_list

class SynthList(SynthParagraph):
    def __init__(self, config):
        super().__init__(config)
    
    @property
    def paragraph(self):
        pass

    @property
    def bullet_list(self):
        lb_sentence, ub_sentence = self.config['sentence_length']
        n_bullet = random_integer_from_list(self.config['n_bullet'])
        seperator = [',', '.', '!']
        items = []
        for i in range(n_bullet):
            n_sentenses = random_integer_from_list(self.config['n_sentences'])
            all_words = [self._gen_random_sentence([lb_sentence, ub_sentence]) for _ in range(n_sentenses)]+ ['']
            item = '&bull' + np.random.choice(seperator).join(all_words)
            items.append(item)
        text = '<br />\n'.join(items)
        title_style = ParagraphStyleGenerator(self.config).style()
        #title_style.name = 'list'
        return Paragraph(text, title_style)    
