import numpy as np 
from paragraph import SynthParagraph
from reportlab.platypus import Paragraph
from style_generator import ParagraphStyleGenerator

class SynthTitle(SynthParagraph):
    def __init__(self, config):
        super().__init__(config)
    
    @property
    def paragraph(self):
        pass

    @property
    def title(self):
        lb_sentence, ub_sentence = self.config['sentence_length']
        all_words = [self._gen_random_sentence([lb_sentence, ub_sentence]) for _ in range(self.config['n_lines'][0], self.config['n_lines'][1])]
        text = '<br />\n'.join(all_words)
        title_style = ParagraphStyleGenerator(self.config).style()
        return Paragraph(text, title_style)    


class SynthSubTitle(SynthParagraph):
    def __init__(self, config):
        super().__init__(config)
    
    @property
    def paragraph(self):
        pass

    @property
    def subtitle(self):
        cn_leading = list('一二三四五六七八九十')
        lb_sentence, ub_sentence = self.config['sentence_length']
        word = self._gen_random_sentence([lb_sentence, ub_sentence]) 
        text = '、'.join([np.random.choice(cn_leading), word])
        subtitle_style = ParagraphStyleGenerator(self.config).style()
        subtitle_style.alignment = 0
        return Paragraph(text, subtitle_style)    
