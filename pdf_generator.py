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
from utils import pdf2img
import matplotlib.pyplot as plt 


pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))


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
    def __init__(self, filename = 'test.pdf'):
        self.filename = filename
        self.doc = myTemplate(filename, pagesize = A4, bottomup = 0, showBoundary = 0, leftMargin = 72)
    @property
    def elements(self):
        if not hasattr(self, '_elements'):
            self._elements = [] 
        return self._elements
    
    def add_table(self):
        nrows = np.random.randint(4, 15)
        ncols = np.random.randint(2, 8)
        grid = True if np.random.random() > 0.7 else False
        table = SynthTable(nrows = nrows, ncols = ncols, grid = grid).table
        self.elements.append(table)

    def add_paragraph(self):
        n_sentenses = np.random.randint(5, 50)
        parag = SynthParagraph(n_sentenses = n_sentenses).paragraph
        self.elements.append(parag)

    def add_spacer(self):
        W, H = A4
        spacer = Spacer(W, 24)
        self.elements.append(spacer)
        
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
    def __init__(self, n_sentenses = 30):
        self.n_sentenses = n_sentenses
        self.paragraph_style = ParagraphStyle(fontName = 'SimSun', fontSize = 12, name = 'Song', firstLineIndent = 24, alignment = TA_JUSTIFY)
        

    @property
    def paragraph(self):
        #seperator = [', ', '，', ': ', '： ', '. ', '。', '! ', '！ ', '? ', '？ ']
        seperator = [',', '.', '!']
        all_words = [self._gen_random_sentence() for _ in range(self.n_sentenses)]
        text = ''
        for w in all_words:
            text += w
            text += np.random.choice(seperator)
        return Paragraph(text, self.paragraph_style)
    
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


        
class SynthTable:
    fileColName = 'data/colName'
    fileColValue = 'data/colValue'
    fileColItem = 'data/colItem'
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    cn_char_cache = []
    
    def __init__(self, nrows = 5, ncols = 4, grid = True, N = 1, random = True):
        self.num = N
        self.nrows = nrows
        self.ncols = ncols 
        self.grid = grid
        self.random = random
        self.grid_table_style =  TableStyle([ ('FONT', (0,0), (-1,-1), 'SimSun'),
                                              ('ALIGN',(0,0),(-1,-1),'CENTER'),
                                              ('FONTSIZE', (0,0), (-1,-1), 12),
                                              ('GRID', (0,0), (-1,-1), 1, colors.black)])
        self.gridless_table_style = [('FONT', (0,0), (-1,-1), 'SimSun'),
                                     ('FONTSIZE', (0,0), (-1,-1), 12),] 

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
        return [self._gen_random_word([min_len, max_len]) for _ in range(self.nrows)]

    def _gen_random_col_header(self, min_len = 2, max_len = 6):
        return [self._gen_random_word([min_len, max_len]) for _ in range(self.ncols)]
        
    def _gen_random_content(self, num, kind = 'number', n_digits = 2, n_integer = 6):
        numbers = list(range(0, 10)) 
        all_digits = [np.random.choice(numbers, n_digits) for _ in range(num)]
        all_digits = [''.join([str(x) for x in digit]) for digit in all_digits]
        all_ints = np.random.randint(10**n_integer, 10**(n_integer + 1), num).tolist()
        return ['.'.join([str(x), str(y)]) for x, y in zip(all_ints, all_digits)]

    def _gen_table_content(self, colHeader = None, rowHeader = None, tableContent = None):
        if colHeader is None:
            colHeader = self._gen_random_col_header()
        if rowHeader is None:
            rowHeader = self._gen_random_row_header()
        if tableContent is None:
            tableContent = self._gen_random_content((self.ncols - 1) * (self.nrows - 1))
            
        table_data = []
        content_ptr = 0
        for i in range(self.nrows):
            if i == 0:
                table_data.append(colHeader[:self.ncols])
            else:
                table_data.append([rowHeader.pop()] + tableContent[content_ptr: content_ptr + self.ncols -1])
                content_ptr += self.ncols  - 1
        style = self.grid_table_style if self.grid else self.gridless_table_style
        table = Table(table_data, style = style)
        return table

# ======================= run ============================

#tb = SynthTable(nrows = 5, ncols = 3)
#print(tb.table)

# pa = SynthParagraph()
# print(pa.paragraph)

sp = SynthPage()
sp.add_spacer()
sp.add_paragraph()
sp.add_spacer()
sp.add_table()
sp.add_spacer()
sp.add_paragraph()
sp.add_spacer()
sp.add_paragraph()
sp.add_spacer()
sp.add_table()
sp.add_spacer()

sp.as_pdf()
sp.as_img()
sp.annotate(show = False)

