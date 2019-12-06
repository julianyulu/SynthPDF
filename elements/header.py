import numpy as np
from reportlab.pdfgen import canvas
from elements.utils import random_integer_from_list
from elements.style_generator import ParagraphStyleGenerator

class PageHeader:
    """
    config: config for the header only 
    """
    _cn_char_cache = []
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    
    def __init__(self, config):
        self.config = config
        
    @property
    def cnChar(self):
        if not self._cn_char_cache:
            with open(self.CN_CHAR_FILE, 'r') as fid:
                content = fid.readlines()
            self._cn_char_cache = [x.strip() for x in content]
        return self._cn_char_cache

    def _gen_line_coords(self):
        cfg_line = self.config['line']
        left = random_integer_from_list(cfg_line['left_margin'])
        top = random_integer_from_list(cfg_line['top_margin'])
        linewidth = random_integer_from_list(cfg_line['linewidth'])
        
        if cfg_line['center']:
            right = left
        else:
            right = random_integer_from_list(cfg_line['right_margin'])
        return left, right, top, linewidth

    def _gen_text(self):
        cfg_text = self.config['text']
        word_length = cfg_text['word_length']
        word_len = random_integer_from_list(word_length)                
        word = ''.join(np.random.choice(self.cnChar, size = word_len, replace = True).tolist())
        return word 
        
    def __call__(self, canvas, doc):
        canvas.saveState()
        w, h = canvas._pagesize
        
        # Gen hline coords 
        left_margin, right_margin, top_margin, linewidth = self._gen_line_coords()
        line_start_x = left_margin
        line_start_y = h - top_margin
        line_end_x = w - right_margin
        line_end_y = h - top_margin  

        # Set text fonts 
        font_name = ParagraphStyleGenerator._gen_font()
        font_size = random_integer_from_list(self.config['text']['font_size'])
        canvas.setFont(font_name, font_size)

        # draw text 
        text_locations = self.config['text']['locations']
        random_locations = np.random.choice(text_locations, size = np.random.randint(0, len(text_locations) + 1), replace = False)
        for loc in random_locations:
            if loc == 'left':
                canvas.drawString(line_start_x, line_start_y + font_size // 3, self._gen_text())
            elif loc == 'right':
                words = self._gen_text()
                canvas.drawString(line_end_x - font_size * len(words), line_start_y + font_size // 3, words)
            elif loc == 'center':
                canvas.drawCentredString(w/2, line_start_y + font_size // 3, self._gen_text())
            else:
                raise ValueError("Text location %s not identified !"%loc)
   
        # draw hline 
        canvas.setLineWidth(linewidth)
        canvas.line(line_start_x, line_start_y, line_end_x, line_end_y)

        # add footer
        line_y = random_integer_from_list([20, 40])
        canvas.drawCentredString(w/2, line_y, str(np.random.randint(100)))
        
        canvas.restoreState()
