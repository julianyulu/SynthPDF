from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, BaseDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY

from reportlab.lib.pagesizes import inch
from reportlab.pdfbase.pdfmetrics import stringWidth

import os
import cv2 
import numpy as np
from utils import pdf2img, load_yaml
import matplotlib.pyplot as plt 

from style_generator import TableStyleGenerator, ParagraphStyleGenerator

pdfmetrics.registerFont(TTFont('SimSun', 'fonts/SimSun.ttf'))


class myTemplate(SimpleDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, **kw)
        self.coords = []

    def afterFlowable(self, flowable):
        x_lowerLeft, y_lowerLeft = self.frame._x, self.frame._y
        x_upperRight, y_upperRight = self.frame._x2, self.frame._y2
        x_lowerLeft = x_lowerLeft - self.frame._leftPadding 

        if isinstance(flowable, Paragraph):
            kind = 'paragraph'
            width, height = flowable.width, flowable.height
        elif isinstance(flowable, Table):
            kind = 'table'
            width, height = flowable._width, flowable._height
        # elif isinstance(flowable, Spacer):
        #     kind = 'spacer'
        #     width, height = flowable.width, flowable.height
        else:
            return -1
        
        x_lowerLeft = (x_upperRight + x_lowerLeft) / 2 - width / 2
        result =  {'kind':  kind,
                   'x': x_lowerLeft,
                   'y': y_lowerLeft, 
                   'w': width,
                   'h': height}
        self.coords.append(result)


class SynthPage:
    def __init__(self, config, filename = 'test.pdf'):
        self.filename = filename
        self.config = config
        self.doc = myTemplate(filename, pagesize = A4, bottomup = 0, showBoundary = 0, leftMargin = 72)
    @property
    def elements(self):
        if not hasattr(self, '_elements'):
            self._elements = [] 
        return self._elements
    
    def add_table(self):
        table = SynthTable(self.config['table']).table
        self.elements.append(table)

    def add_paragraph(self):
        n_sentenses = np.random.randint(5, 50)
        parag = SynthParagraph(self.config['paragraph']).paragraph
        self.elements.append(parag)

    def add_spacer(self):
        W, H = A4
        height = self.config['spacer']['height']
        if len(height) == 1:
            h = height[0]
        elif len(height) == 2:
            h = np.random.randint(height[0], height[1] + 1)
        else:
            h = np.random.choice(height)
        spacer = Spacer(W, h)
        self.elements.append(spacer)

    def add_title(self):
        title = SynthTitle(self.config['title']).title
        self.elements.append(title)
        
    def as_pdf(self):
        self.doc.build(self.elements)

    def as_img(self):
        if not os.path.isfile(self.filename): self.to_pdf
        if not hasattr(self, '_img_files'): 
            image_path = pdf2img(self.filename)
            self._img_files = image_path
        return self._img_files 

    def _trans_coords(self, coord, W, H):
        """
        Translate pdf coords start from lower-left coorner to uppper left coord on output image
        coords: (x, y) 
        """
        x, y = coord
        W_A4, H_A4 = A4
        factor = H / H_A4 
        return (int(x * factor), int(H - y * factor))
    
    def annotate(self, show = False):
        
        annot = self.doc.coords

        # current only annote the first page, as the coords has not info about the page
        img_files = self.as_img()
        img = cv2.imread(img_files[0])
        H, W = img.shape[:2]

        prev_ycoords = -1  # filter out the annot on the second page
        for elem in annot:
            x_lowerLeft, y_lowerLeft = elem['x'], elem['y']
            w_elem, h_elem = elem['w'], elem['h']
            x_upperLeft, y_upperLeft = self._trans_coords((x_lowerLeft, y_lowerLeft + h_elem), W, H)
            x_lowerRight, y_lowerRight = self._trans_coords((x_lowerLeft + w_elem, y_lowerLeft), W, H)

            if y_upperLeft > prev_ycoords:
                prev_ycoords = y_upperLeft
                cv2.rectangle(img,
                              (x_upperLeft, y_upperLeft),
                              (x_lowerRight, y_lowerRight),
                              255, 2)
            else:
                break 
        save_name = img_files[0].split('.jpg')[0] + '_ann.jpg'
        cv2.imwrite(save_name, img)
        if show:
            plt.figure(figsize = (12, 20))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        
class SynthParagraph:
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    cn_char_cache = []
    def __init__(self, config):
        self.config = config
        self.n_sentences  = config['n_sentences']
        
    @property
    def n_sentences(self):
        return self._n_sentences

    @n_sentences.setter
    def n_sentences(self, num):
        if len(num) == 1:
            self._n_sentences = num[0]
        elif len(num) == 2:
            self._n_sentences = np.random.choice(range(num[0], num[1] + 1))
        else:
            self._n_sentences = np.random.choice(num)
        
        
    @property
    def paragraph(self):
        #seperator = [', ', '，', ': ', '： ', '. ', '。', '! ', '！ ', '? ', '？ ']
        seperator = [',', '.', '!']
        lb_sentence, ub_sentence = self.config['sentence_length']
        all_words = [self._gen_random_sentence([lb_sentence, ub_sentence]) for _ in range(self._n_sentences)]
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
        if type(length) == list:
            word_len = np.random.choice(np.arange(length[0], length[1] + 1))
        else:
            word_len = length
                        
        word = ''.join(np.random.choice(self.cnChar, size = word_len, replace = True).tolist())
        return word

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
    
class SynthTable:
    fileColName = 'data/colName'
    fileColValue = 'data/colValue'
    fileColItem = 'data/colItem'
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    cn_char_cache = []
    
    def __init__(self, config):
        self.config = config 
        self.nrows = config['layout']['nrows']
        self.ncols = config['layout']['ncols']

    @property
    def ncols(self):
        return self._ncols

    @ncols.setter
    def ncols(self, cols):
        if len(cols) == 1:
            self._ncols == cols[0]
        elif len(cols) == 2:
            self._ncols = np.random.choice(range(cols[0], cols[1] + 1))
        else:
            self._ncols = np.random.choice(cols)

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, rows):
        if len(rows) == 1:
            self._nrows == rows[0]
        elif len(rows) == 2:
            self._nrows = np.random.choice(range(rows[0], rows[1] + 1))
        else:
            self._nrows = np.random.choice(rows)
            
    @property
    def cnChar(self):
        if not self.cn_char_cache:
            with open(self.CN_CHAR_FILE, 'r') as fid:
                content = fid.readlines()
            self.cn_char_cache = [x.strip() for x in content]
        return self.cn_char_cache

    @property
    def table(self):
        return self._gen_table_content()
    
    def _gen_random_word(self, length = [2, 6], random_length = False):
        if type(length) == list:
            word_len = np.random.choice(np.arange(length[0], length[1] + 1))
        else:
            word_len = length
                        
        word = ''.join(np.random.choice(self.cnChar, size = word_len, replace = True).tolist())
        return word

    def _gen_random_row_header(self, min_len = 2, max_len = 6):
        return [self._gen_random_word([min_len, max_len]) for _ in range(self._nrows)]

    def _gen_random_col_header(self, min_len = 2, max_len = 6):
        return [self._gen_random_word([min_len, max_len]) for _ in range(self._ncols)]
        
    def _gen_random_content(self, num, kind = 'number', n_digits = 2, n_integer = 6):
        numbers = list(range(0, 10)) 
        all_digits = [np.random.choice(numbers, n_digits) for _ in range(num)]
        all_digits = [''.join([str(x) for x in digit]) for digit in all_digits]
        all_ints = np.random.randint(10**n_integer, 10**(n_integer + 1), num).tolist()
        return ['.'.join([str(x), str(y)]) for x, y in zip(all_ints, all_digits)]

    def _gen_table_space(self):
        space_before = self._gen_random_int_from_list(self.config['layout']['space_before'])
        space_after = self._gen_random_int_from_list(self.config['layout']['space_after'])
        return space_before, space_after

    def _gen_random_int_from_list(self, inlist):
        if len(inlist) == 1:
            return inlist[0]
        elif len(inlist) == 2:
            return np.random.randint(inlist[0], inlist[1])
        else:
            return np.random.choice(inlist)
        
    def _gen_table_content(self, colHeader = None, rowHeader = None, tableContent = None):
        if colHeader is None:
            colHeader = self._gen_random_col_header()
        if rowHeader is None:
            rowHeader = self._gen_random_row_header()
        if tableContent is None:
            tableContent = self._gen_random_content((self._ncols - 1) * (self._nrows - 1))
            
        table_data = []
        content_ptr = 0
        for i in range(self._nrows):
            if i == 0:
                table_data.append(colHeader[:self._ncols])
            else:
                table_data.append([rowHeader.pop()] + tableContent[content_ptr: content_ptr + self._ncols -1])
                content_ptr += self._ncols  - 1

        space_before, space_after = self._gen_table_space()
        style = TableStyleGenerator(self.config).style()
        table = Table(table_data,
                      style = style,
                      spaceBefore = space_before,
                      spaceAfter = space_after)
        return table


class PageMixer:
    def __init__(self, config):
        self.config = config
        self.page = SynthPage(config) 
        self.elements = [x for x in dir(self.page) if x.startswith('add_')]

    def _make_prob_book(self):
        mixer_config = self.config['mixer']
        prob_book = {x.split('prob_')[1]: mixer_config[x] for x in mixer_config if 'prob_' in x}
        sum_prob = sum(prob_book.values())
        acc_prob = 0
        for key in prob_book:
            prob_book[key] = [acc_prob, prob_book[key] / sum_prob]
            acc_prob += prob_book[key][1]
        return prob_book

    def make_single(self):
        prob_book = self._make_prob_book()
        max_elements = self.config['mixer']['max_elements_per_page']
        lb_tables = self.config['mixer']['min_tables_per_page']
        ub_tables = self.config['mixer']['max_tables_per_page']
        n_tables = np.random.randint(lb_tables, ub_tables + 1)

        select_elements = ['add_table'] * n_tables
        n_elements = len(select_elements)        
        while n_elements < max_elements:
            elem = np.random.choice(self.elements)
            select_elements.append(elem)
            n_elements += 1
        np.random.shuffle(select_elements)

        for op in select_elements:
            self.page.__getattribute__(op)()

        if self.config['mixer']['as_pdf']:
            self.page.as_pdf()

        if self.config['mixer']['as_img']:
            self.page.as_img()

        if self.config['mixer']['annotate']:
            self.page.annotate()

# ======================= run ============================

#tb = SynthTable(nrows = 5, ncols = 3)
#print(tb.table)

# pa = SynthParagraph()
# printa.paragraph)

config =  load_yaml('config.yaml')
m = PageMixer(config)
m.make_single()
# sp = SynthPage(config)
# sp.add_spacer()
# sp.add_title()
# sp.add_paragraph()
# sp.add_spacer()
# sp.add_table()
# sp.as_pdf()
# sp.as_img()
# sp.annotate(show = False)

