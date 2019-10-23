import os
import cv2
import json
import uuid 
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt 

from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table
from utils import pdf2img, load_yaml, random_integer_from_list, prob2category
from style_generator import TableStyleGenerator, ParagraphStyleGenerator


class myTemplate(SimpleDocTemplate):
    def __init__(self, filename, config, **kw):
        super().__init__(filename, **kw)
        self.config = config
        self.coords = []

    def afterFlowable(self, flowable):
        x_lowerLeft, y_lowerLeft = self.frame._x, self.frame._y
        x_upperRight, y_upperRight = self.frame._x2, self.frame._y2
        page_number = self.canv._pageNumber  # start from 1 
        #print(type(flowable))
        #print(self.canv._pageNumber)
        #print(flowable.__dict__)
        #print(self.frame.__dict__)
        # parse flowable coords
        if isinstance(flowable, Paragraph):
            kind = 'paragraph'
            width, height = flowable.width, flowable.height # note the difference from table 
        elif isinstance(flowable, Table):
            kind = 'table'
            width, height = flowable._width, flowable._height
                        
        # elif isinstance(flowable, Spacer):
        #     kind = 'spacer'
        #     width, height = flowable.width, flowable.height
        else:
            return -1
        
        # fix shifts
        x_lowerLeft = x_lowerLeft - self.frame._leftPadding
        x_lowerLeft = (x_upperRight + x_lowerLeft) / 2 - width / 2
        y_lowerLeft = y_lowerLeft + self.frame._prevASpace 

        # add flowable result to coords 
        result =  {'kind':  kind,
                   'page': page_number, 
                   'is_flowable': True, 
                   'x': x_lowerLeft,
                   'y': y_lowerLeft, 
                   'w': width,
                   'h': height}
        self.coords.append(result)

        # Parse special none flowables
        
        ## stamp at bordered table corners
        ## only table has attribute '_linecmds' == [.....]
        ## Bordered table is a none empty list, while borderless table is [] 
        if hasattr(flowable, '_linecmds') and flowable._linecmds and np.random.random() < self.config['stamp']['prob']:
            x_shift = random_integer_from_list(self.config['stamp']['corner_dx'])
            y_shift = random_integer_from_list(self.config['stamp']['corner_dy'])
            # choose one of the 4 corners and add shift
            stamp_width = random_integer_from_list(self.config['stamp']['width'])
            x_stamp = int(np.random.choice([x_lowerLeft, x_lowerLeft + width]) - stamp_width / 2 + x_shift)
            y_stamp = int(np.random.choice([y_lowerLeft, y_lowerLeft + height]) - stamp_width / 2 + y_shift)
            
            stamp_path = self.config['stamp']['stamp_img_path']
            stamp_file = np.random.choice(os.listdir(stamp_path))
            
            img_width, img_height = self.canv.drawImage(os.path.join(stamp_path, stamp_file),
                                                        x_stamp, y_stamp, mask = 'auto',
                                                        anchor = 'c', # anchored at center 
                                                        width = stamp_width, height = stamp_width)
                
            self.coords.append({'kind': 'stamp',
                                'is_flowable': False,
                                'page': page_number, 
                                'x': x_stamp,
                                'y': y_stamp,
                                'w': stamp_width,
                                'h': stamp_width})
                
class SynthPage:
    def __init__(self, config, filename = 'test.pdf'):
        self.filename = filename
        self.config = config
        self._initialize()
        self.doc = myTemplate(self._pdf_file,
                              config['notFlowables'],
                              pagesize = A4, bottomup = 1,
                              showBoundary = 0, leftMargin = 72)
        

    def _initialize(self):
        output_path = self.config['io']['output_path']
        if not os.path.exists(output_path): os.mkdir(output_path)
        self._output_path = output_path
        
        pdf_path = os.path.join(output_path, self.config['io']['pdf_path'])
        if not os.path.exists(pdf_path): os.mkdir(pdf_path)
        self._pdf_path = pdf_path

        img_path = os.path.join(output_path, self.config['io']['img_path'])
        if not os.path.exists(img_path): os.mkdir(img_path)
        self._img_path = img_path

        json_path = os.path.join(output_path, self.config['io']['json_path'])
        if not os.path.exists(json_path): os.mkdir(json_path)
        self._json_path = json_path

        self._pdf_file = os.path.join(pdf_path, self.filename)
                                
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
        h = random_integer_from_list(height)
        spacer = Spacer(W, h)
        self.elements.append(spacer)

    def add_title(self):
        title = SynthTitle(self.config['title']).title
        self.elements.append(title)
        
    def as_pdf(self):
        self.doc.build(self.elements)

    def as_img(self):
        if not os.path.isfile(self._pdf_file): self.as_pdf
        if not hasattr(self, '_img_files'): 
            image_path = pdf2img(os.path.join(self._pdf_path, self.filename),
                                 save_path = self._img_path)
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

    
    def annotate(self, show_img = False, save_img = False, save_json = False):
        
        annot = self.doc.coords
        # current only annote the first page, as the coords has not info about the page
        img_files = self.as_img()
        img = cv2.imread(img_files[0])
        H, W = img.shape[:2]

        labels = {}
        idx = 0 
        prev_ycoords = -1  # filter out the annot on the second page
        for elem in annot:
            x_lowerLeft, y_lowerLeft = elem['x'], elem['y']
            w_elem, h_elem = elem['w'], elem['h']
            x_upperLeft, y_upperLeft = self._trans_coords((x_lowerLeft, y_lowerLeft + h_elem), W, H)
            x_lowerRight, y_lowerRight = self._trans_coords((x_lowerLeft + w_elem, y_lowerLeft), W, H)

            if elem['page'] == 1:
                #prev_ycoords = y_upperLeft
                labels[idx] = {'kind': elem['kind'],
                               'p1': (int(x_upperLeft), int(y_upperLeft)),
                               'p2': (int(x_lowerRight), int(y_lowerRight))}
                idx += 1 
                if show_img or save_img:
                    cv2.rectangle(img,
                                  (x_upperLeft, y_upperLeft),
                                  (x_lowerRight, y_lowerRight),
                                  255, 2)
            else:
                break
                
        if save_img:
            save_name = os.path.basename(img_files[0]).split('_page1.jpg')[0] + '_ann.jpg'
            cv2.imwrite(os.path.join(self._img_path, save_name), img)
        
        if show_img:
            plt.figure(figsize = (12, 20))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

        if save_json:
            save_name = os.path.basename(img_files[0]).split('_page1.jpg')[0] + '.json'
            with open(os.path.join(self._json_path, save_name), 'w+') as fid:
                json.dump(labels, fid)
        
        return labels # dicts
    
class SynthParagraph:
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    cn_char_cache = []
    def __init__(self, config):
        self.config = config
        
    @property
    def paragraph(self):
        #seperator = [', ', '，', ': ', '： ', '. ', '。', '! ', '！ ', '? ', '？ ']
        seperator = [',', '.', '!']
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
        self._ncols = random_integer_from_list(cols)

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, rows):
        self._nrows = random_integer_from_list(rows)
            
    @property
    def cnChar(self):
        if not hasattr(self, '_cn_char_cache'):
            with open(self.CN_CHAR_FILE, 'r') as fid:
                content = fid.readlines()
            self._cn_char_cache = [x.strip() for x in content]
        return self._cn_char_cache

    @property
    def enChar(self):
        if not hasattr(self, '_en_char_cache'):
            content = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            self._en_char_cache = content
        return self._en_char_cache

    @property
    def cnSymbol(self):
        if not hasattr(self, '_cn_symbol_cache'):
            symbols = list("，。、：！？")
            self._cn_symbol_cache = symbols
        return self._cn_symbol_cache

    @property
    def enSymbol(self):
        if not hasattr(self, '_en_symbol_cache'):
            symbols = list(",.?!()")
            self._en_symbol_cache = symbols
        return self._en_symbol_cache
    
    @property
    def table(self):
        cfg_blocks = self.config['content']['blocks']
        cfg_number = self.config['content']['numbers']
        cfg_cn = self.config['content']['cn_chars']
        cfg_en = self.config['content']['en_chars']
        cfg_special = self.config['content']['special']

        # First generate row/col headers  [list of n_col or n_row elements
        row_header = self._gen_cell_content(cfg_blocks['row_header'],  self.nrows)
        col_header = self._gen_cell_content(cfg_blocks['col_header'],  self.ncols)

        # Then generate table contents [ list of (col-1)*(row-1) elements
        content = self._gen_cell_content(cfg_blocks['content'],  (self.ncols - 1) * (self.nrows - 1))

        #print(self.nrows, self.ncols, '\n', content)
        # Then add special docorations
        #prob_underline = cfg_special['prob_underline']
        prob_parentheses = cfg_special['prob_parentheses']
        prob_empty = cfg_special['prob_empty']
        prob_dash = cfg_special['prob_dash'] 
        content = self._decorate_parentheses(content, prob_parentheses)
        #content = self._decorate_underline(content, prob_underline)
        content = self._decorate_empty(content, prob_empty)
        content = self._decorate_dash(content, prob_dash)

        # Then merge content with headers to a [n_rows x n_cols] list
        content_ptr = 0
        table_data = [] 
        for i in range(self.nrows):
            if i == 0:
                table_data.append(col_header[:self._ncols])
            else:
                table_data.append([row_header.pop()] + content[content_ptr: content_ptr + self._ncols -1])
                content_ptr += self._ncols  - 1

        # Then add random empty col 
        if np.random.random() < self.config['space']['prob_empty_col'] and 3<= self._ncols <=5:
            empty_col = 1 if np.random.random() < 0.5 else np.random.randint(1, self._ncols-1)
            empty_size = random_integer_from_list(self.config['space']['size_empty_col'])
            for i in range(len(table_data)):
                table_data[i][empty_col] = ' ' * empty_size
                
        # Finally build the table instance
        space_before, space_after = self._gen_table_space()
        style = TableStyleGenerator(self.config).style()
        table = Table(table_data,
                      style = style,
                      spaceBefore = space_before,
                      spaceAfter = space_after)
        return table
    
    def _gen_random_cn_sentence(self, length = [2, 6]):
        word_len = random_integer_from_list(length)
        sentence = ''.join(np.random.choice(self.cnChar, size = word_len, replace = True).tolist())
        return sentence

    def _gen_random_en_word(self, length = [2, 6]):
        word_len = random_integer_from_list(length)
        word = ''.join(np.random.choice(self.enChar, size = word_len, replace = True).tolist())
        return word
    
    def _gen_random_cn_symbol(self):
        return np.random.choice(self.cnSymbol)
    
    def _gen_random_en_symbol(self):
        return np.random.choice(self.enSymbol)
    
    def _gen_random_decimal(self, n_integers = [4], n_digits = [2]):
        n_int = random_integer_from_list(n_integers)
        n_dig = random_integer_from_list(n_digits)
        part_int = ''.join(np.random.choice(list('0123456789'), size = n_int, replace = True))
        part_dig = ''.join(np.random.choice(list('0123456789'), size = n_dig, replace = True))
        if len(part_dig) == 0:
            return part_int
        elif len(part_int) == 0:
            part_int = '0'
            return '.'.join([part_int, part_dig])
        else:
            return '.'.join([part_int, part_dig])

    def _gen_random_percents(self, n_integers = [4], n_digits = [2]):
        part_number = self._gen_random_decimal(n_integers, n_digits)
        return ''.join([part_number, '%'])

    # def _decorate_underline(self, source_list, prob):
    #     dst_list = ['<p><u>' + s + '</u></p>' if np.random.random() <= prob else s for s in source_list]
    #     return dst_list

    def _decorate_parentheses(self, source_list, prob):
        dst_list = ['(' + s + ')' if np.random.random() <= prob else s for s in source_list]
        return dst_list 

    def _decorate_empty(self, source_list, prob):
        dst_list = [' ' if np.random.random() <= prob else s for s in source_list]
        return dst_list

    def _decorate_dash(self, source_list, prob):
        dst_list = ['-' if np.random.random() <= prob else s  for s in source_list]
        return dst_list
    
    def _gen_cell_content(self, sub_block_config, N):
        prob2item = prob2category(sub_block_config)
        result = [] 
        for _ in range(N):
            rand = np.random.random()
            category = prob2item(rand)
            if 'number' in category:
                res = self._gen_random_decimal(self.config['content']['numbers']['n_integers'],
                                         self.config['content']['numbers']['n_digits'])
            elif 'cn_char' in category:
                res = self._gen_random_cn_sentence(self.config['content']['cn_chars']['word_length'])
            elif 'en_char' in category:
                res = self._gen_random_en_word(self.config['content']['en_chars']['word_length'])
            elif '_percent' in category:
                res = self._gen_random_percents(self.config['content']['numbers']['n_integers'],
                                                self.config['content']['numbers']['n_digits'])
            else:
                raise ValueError("category not recognized: %s"%category)
            result.append(res)
        return result

    def _gen_table_space(self):
        space_before = random_integer_from_list(self.config['layout']['space_before'])
        space_after = random_integer_from_list(self.config['layout']['space_after'])
        return space_before, space_after


class PageMixer:
    def __init__(self, config, filename = 'test.pdf'):
        self.config = config
        self.page = SynthPage(config, filename = filename) 
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

    def make(self):
        prob_book = self._make_prob_book()
        max_elements = self.config['mixer']['max_elements_per_page']
        lb_tables = self.config['mixer']['min_tables_per_page']
        ub_tables = self.config['mixer']['max_tables_per_page']
        n_tables = np.random.randint(lb_tables, ub_tables + 1)

        # fist gen random number table elements 
        select_elements = ['add_table'] * n_tables
        n_elements = len(select_elements)        
        while n_elements < max_elements:
            elem = np.random.choice(self.elements)
            select_elements.append(elem)
            n_elements += 1
        np.random.shuffle(select_elements)

        # then double check if avoid neighbor tables
        if self.config['mixer']['avoid_neighbor_tables']:
            prev = ''
            for i in range(len(select_elements)):
                if '_table' in select_elements[i] and '_table' in prev:
                    select_elements[i] = 'add_paragraph'
                prev = select_elements[i]
        # add elements to page from select_elements 
        for op in select_elements:
            self.page.__getattribute__(op)()

        # finally output result file 
        if self.config['mixer']['as_pdf']:
            self.page.as_pdf()

        if self.config['mixer']['as_img']:
            self.page.as_img()

        if self.config['mixer']['annotate']:
            annot = self.page.annotate(save_img = self.config['mixer']['save_annotate_imgs'],
                                       show_img = self.config['mixer']['show_annotate_imgs'],
                                       save_json = self.config['mixer']['save_single_annotate_json'])
        return annot 
    


if __name__ == '__main__':
    config =  load_yaml('config.yaml')
    cfg_runner = config['runner']

    run_parallel = cfg_runner['run_parallel']
    n_files = cfg_runner['n_files']
    n_processors = cfg_runner['num_processors']

    if run_parallel:
        
        filenames = [uuid.uuid4().hex + '.pdf' for _ in range(n_files)]
        SynthPage(config)._initialize() # make output folders to avoid parallel conflict 
        def runner(filename):
            _ = PageMixer(config, filename).make()
        with Pool(n_processors) as p:
            _ = list(tqdm(p.imap_unordered(runner, filenames), total = n_files))
    else:
        for _ in tqdm(range(n_files)):
            filename = uuid.uuid4().hex + '.pdf'
            m = PageMixer(config, filename)
            m.make()
    
# ======================= test run ============================

#config =  load_yaml('config.yaml')

#Test Table 
#tb = SynthTable(config['table'])
#print(tb.nrows, tb.ncols)
#print(tb.table)

## Test Paragraph
# pa = SynthParagraph()
# printa.paragraph)

# ##Test SynthPage
# sp = SynthPage(config)
# sp.add_paragraph()
# sp.add_table()
# sp.add_paragraph()
# sp.add_table()
# sp.add_title()
# sp.as_pdf()
# sp.as_img()
# sp.annotate(save_img = True)

## Test Mixer 
# m = PageMixer(config)
# m.make()
#print(m.page.doc.coords)
