import os
import numpy as np
from utils import load_yaml

from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT

FONT_PATH = 'fonts'

class TableStyleGenerator:
    def __init__(self, config):
        self.config = config

    def style(self):
        styles = []
        for x in dir(self):
            if '_gen_' in x:
                elem_style = self.__getattribute__(x)()
                if elem_style:
                    styles.append(elem_style)
        return TableStyle(styles)

    ## ====================== CELL ==========================
    def _gen_font(self):
        font_file = np.random.choice(os.listdir(FONT_PATH))
        font_name = font_file.split('.ttf')[0] 
        pdfmetrics.registerFont(TTFont(font_name, font_file))
        style = ('FONT', (0, 0), (-1, -1), font_name)
        return style

    def _gen_font_size(self):
        lb, ub  = self.config['cell']['font_size']   # lower / upper bounds of font size
        size = np.random.randint(lb, ub + 1)
        style = ('FONTSIZE', (0, 0), (-1, -1), size)
        return style

    def _gen_font_color(self):
        font_color = self.config['cell']['text_color']
        if len(font_color) == 1:
            color = font_color[0]
        elif len(font_color) == 2:
            pass
            # TODO: random color in a given rbg range 
        else:
            color = np.random.choice(font_color)
        style = ('TEXTCOLOR', (0, 0), (-1, -1), color)
        return style

    ## does not work 
    
    # def _gen_vert_align(self):
    #     align_candi = self.config['cell']['vertical_align']
    #     if len(align_candi) == 0:
    #         candidates = ['TOP', 'MIDDLE', 'BOTTOM']
    #         align = np.random.choice(candidates)
    #     elif len(align_candi) == 1:
    #         align = align_candi[0]
    #     else:
    #         align = np.random.choice(align_candi)
    #     style = ('VALIGN', (0, 0), (-1, -1), align)
    #     return style

    def _gen_hori_align(self):
        align_candi = self.config['cell']['horizontal_align']
        if len(align_candi) == 0:
            candidates = ['LEFT', 'RIGHT', 'CENTER', 'DECIMAL']
            align = np.random.choice(candidates)
        elif len(align_candi) == 1:
            align = align_candi[0]
        else:
            align = np.random.choice(align_candi)
        style = ('ALIGNMENT', (0, 0), (-1, -1), align)
        return style

    ## ====================== LINE ==========================
    def _gen_grid(self):
        grid_color = self.config['line']['grid_color']
        grid_width = self.config['line']['grid_line_width']
        grid_prob = self.config['line']['grid_prob']
        
        has_grid = True if np.random.random() < grid_prob else False
        
        if len(grid_color) == 1:
            color = grid_color[0]
        elif len(grid_color) == 2:
            # TODO: random color in a given rbg range 
            pass
        else:
            color = np.random.choice(grid_color)

        if len(grid_width) == 1:
            line_width = grid_width[0]
        elif len(grid_width) == 2:
            line_width = np.random.choice(range(grid_width[0], grid_width[1] + 1))
        else:
            line_width = np.random.choice(grid_widht)
            
        if has_grid:
            style = ('GRID', (0, 0), (-1, -1), line_width, color)
            return style 
        else:
            return None 

    
class ParagraphStyleGenerator:
    def __init__(self, config):
        self.config = config

    def style(self):
        style = ParagraphStyle(fontName = self._gen_font(), 
                               fontSize = self._gen_font_size(),
                               textColor = self._gen_font_color(),
                               spaceBefore = 100, 
                               name = 'custom',
                               leading = self._gen_leading(),
                               firstLineIndent = 24,
                               alignment = self._gen_align())
        return style
    
    def _gen_font(self):
        font_file = np.random.choice(os.listdir(FONT_PATH))
        font_name = font_file.split('.ttf')[0] 
        pdfmetrics.registerFont(TTFont(font_name, font_file))
        return font_name 

    def _gen_font_size(self):
        if not hasattr(self, '_fontsize'):
            lb, ub  = self.config['font_size']   # lower / upper bounds of font size
            size = np.random.randint(lb, ub + 1)
            self._fontsize = size 
        return self._fontsize

    def _gen_font_color(self):
        font_color = self.config['text_color']
        if len(font_color) == 1:
            color = font_color[0]
        elif len(font_color) == 2:
            pass
            # TODO: random color in a given rbg range 
        else:
            color = np.random.choice(font_color)
        return color
    
    def _gen_leading(self):
        """
        Has to ensure that the leading correspoins to the fontsize in the same run
        """
        leading_factor = self.config['leading']
        
        if len(leading_factor) == 1:
            leading = leading_factor[0] 
        elif len(leading_factor) == 2:
            leading = np.random.random() * (leading_factor[1] - leading_factor[0]) + leading_factor[0]
        else:
            leading = np.random.choice(leading_factor)
            
        if not hasattr(self, '_fontsize'):
            self._gen_font_size()

        leading = leading * self._fontsize
        del self._fontsize
        
        return leading

    def _gen_align(self):
        alignment = self.config['align']
        align_dict = {'justify': TA_JUSTIFY,
                      'center': TA_CENTER,
                      'left': TA_LEFT,
                      'right': TA_RIGHT}
        
        if len(alignment) == 0:
            align = np.random.choice([TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT])
        elif len(alignment) == 1:
            align = align_dict[alignment[0]]
        else:
            align = np.random.choice([align_dict[x] for x in alignment])
        return align
    
        
# config =  load_yaml('config.yaml')
# print(ParagraphStyleGenerator(config['paragraph']).style())
