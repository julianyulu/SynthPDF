import os
import numpy as np
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from elements.utils import load_yaml, random_integer_from_list, prob2category

FONT_PATH = 'fonts'

class TableStyleGenerator:
    def __init__(self, config):
        self.config = config

    def style(self):
        styles = []
        for x in dir(self):
            if '_gen_' in x:
                elem_style = self.__getattribute__(x)()
                if elem_style: styles += elem_style  # list addition
        return TableStyle(styles)

    ## ====================== CELL ==========================
    @staticmethod
    def _gen_font():
        font_file = np.random.choice(os.listdir(FONT_PATH))
        font_name = font_file.split('.ttf')[0] 
        pdfmetrics.registerFont(TTFont(font_name, font_file))
        style = [('FONT', (0, 0), (-1, -1), font_name)]
        return style

    def _gen_font_size(self):
        lb, ub  = self.config['cell']['font_size']   # lower / upper bounds of font size
        size = np.random.randint(lb, ub + 1)
        style = [('FONTSIZE', (0, 0), (-1, -1), size)]
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
        style = [('TEXTCOLOR', (0, 0), (-1, -1), color)]
        return style

    def _gen_hori_align(self):
        align_candi = self.config['cell']['horizontal_align']
        if len(align_candi) == 0:
            candidates = ['LEFT', 'RIGHT', 'CENTER', 'DECIMAL']
            align = np.random.choice(candidates)
        elif len(align_candi) == 1:
            align = align_candi[0]
        else:
            align = np.random.choice(align_candi)
        style = [('ALIGNMENT', (0, 0), (-1, -1), align)]
        return style

    ## ====================== LINE ==========================
    def _gen_grid(self):
        prob_func = prob2category(self.config['line'])
        rand_elem = prob_func(np.random.random())

        if '_no_grid' in rand_elem:
            return None
        else:
            grid_color = self.config['line']['grid_color']
            grid_width = self.config['line']['grid_line_width']
            
            if len(grid_color) == 1:
                color = grid_color[0]
            elif len(grid_color) == 2:
                # TODO: random color in a given rbg range
                raise NotImplementedError("grid_color not implemented !")
            else:
                color = np.random.choice(grid_color)
                
            line_width = random_integer_from_list(grid_width)

            if '_full_grid' in rand_elem:
                style = [('GRID', (0, 0), (-1, -1), line_width, color)]
            elif '_head_hline_2' in rand_elem:
                style = [('LINEABOVE', (0, 0), (-1, 0), line_width, color),
                         ('LINEBELOW', (0, 0), (-1, 0), line_width, color)]
            elif '_head_hline_1'in rand_elem:
                style = [('LINEBELOW', (0, 0), (-1, 0), line_width, color)]
            elif '_hline_only' in rand_elem:
                style = [('LINEABOVE', (0, 0), (-1, 0), line_width, color),
                         ('LINEBELOW', (0, 0), (-1, -1), line_width, color)]
            else:
                raise ValueError("%s not implemented, check your config"%rand_elem)
            return style
        
class ParagraphStyleGenerator:
    def __init__(self, config):
        self.config = config

    def style(self):
        style = ParagraphStyle(fontName = self._gen_font(), 
                               fontSize = self._gen_font_size(),
                               textColor = self._gen_font_color(),
                               spaceBefore = self._gen_space_before(),
                               spaceAfter = self._gen_space_after(), 
                               name = 'text',
                               leading = self._gen_leading(),
                               firstLineIndent = self._gen_firstline_indent(),
                               alignment = self._gen_align())
        return style
    
    @staticmethod 
    def _gen_font():
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

    def _gen_firstline_indent(self):
        indent = self.config['firstline_indent']
        return random_integer_from_list(indent)
    
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
    
    def _gen_space_before(self):
        space_before = self.config['space_before']
        if len(space_before) == 0:
            space = 0 
        elif len(space_before) == 1:
            space = space_before[0]
        elif len(space_before) == 2:
            space = np.random.randint(space_before[0], space_before[1])
        else:
            space = np.random.choice(space_before)
        return space

    def _gen_space_after(self):
        space_after = self.config['space_after']
        if len(space_after) == 0:
            space = 0 
        elif len(space_after) == 1:
            space = space_after[0]
        elif len(space_after) == 2:
            space = np.random.randint(space_after[0], space_after[1])
        else:
            space = np.random.choice(space_after)
        return space
    
# config =  load_yaml('config.yaml')
# print(ParagraphStyleGenerator(config['paragraph']).style())
