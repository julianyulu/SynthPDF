import numpy as np 
from reportlab.platypus import Table, Paragraph
from reportlab.lib.styles import ParagraphStyle
from elements.style_generator import TableStyleGenerator
from elements.utils import (load_yaml,
                            random_integer_from_list,
                            prob2category)
                            

class RotatedTable(Table):
    def wrap(self, availWidth, availHeight):
        h, w = Table.wrap(self, availWidth, availHeight)
        return w, h
    
    def draw(self):
        #self.canv.saveState()
        self.canv.rotate(90)
        Table.draw(self)
        #self.canv.restoreState()



class SynthTable:
    fileColName = 'data/colName'
    fileColValue = 'data/colValue'
    fileColItem = 'data/colItem'
    CN_CHAR_FILE = 'char_data/JianTi3500.txt'
    _cn_char_cache = []
    
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
        if not self._cn_char_cache:
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
        table_style = TableStyleGenerator(self.config).style()
        
        # First generate row/col headers  [list of n_col or n_row elements
        row_header = self._gen_cell_content(cfg_blocks['row_header'],  self.nrows)
        col_header = self._gen_cell_content(cfg_blocks['col_header'],  self.ncols)

        # Then generate table contents [ list of (col-1)*(row-1) elements
        content = self._gen_cell_content(cfg_blocks['content'],  (self.ncols - 1) * (self.nrows - 1))

        
        # Then add special docorations
        prob_parentheses = cfg_special['prob_parentheses']
        prob_empty = cfg_special['prob_empty']
        prob_dash = cfg_special['prob_dash']
        prob_underline = cfg_special['prob_underline']
        content = self._decorate_parentheses(content, prob_parentheses)
        content = self._decorate_underline(content, prob_underline, table_style)
        content = self._decorate_empty(content, prob_empty)
        content = self._decorate_dash(content, prob_dash)

        # Then merge content with headers to a [n_rows x n_cols] list
        content_ptr = 0
        table_data = [] 
        for i in range(self.nrows):
            if i == 0:
                # make the 1st cell empty 
                prob_empty_first_cell = cfg_blocks['prob_empty_first_cell']
                if np.random.random() < prob_empty_first_cell:
                    table_data.append([''] + col_header[1:self._ncols])
                else:
                    table_data.append(col_header[:self._ncols])
            else:
                table_data.append([row_header.pop()] + content[content_ptr: content_ptr + self._ncols -1])
                content_ptr += self._ncols  - 1

        # Then add single random empty col 
        if np.random.random() < self.config['space']['prob_empty_col'] and 3<= self._ncols <=5:
            # set 50% missing cols to be the 2nd col 
            empty_col = 1 if np.random.random() < 0.5 else np.random.randint(1, self._ncols-1)
            empty_size = random_integer_from_list(self.config['space']['size_empty_col'])
            for i in range(len(table_data)):
                table_data[i][empty_col] = ' ' * empty_size

        # Then add single random empty row 
        if np.random.random() < self.config['space']['prob_empty_row'] and 4<= self._nrows:
            empty_row = np.random.randint(1, self._nrows -1) 
            table_data[empty_row] =  [] *self._ncols

        # Then set second row (count start from 1) to be empty: simulate space / gap
        if np.random.random() < self.config['space']['prob_empty_second_row'] and self._nrows>3 and self._ncols > 2:
            table_data[1] = [] * self._ncols

        # Then set last second row (count start from 1) to be empty: simulate space / gap
        if np.random.random() < self.config['space']['prob_empty_last_second_row'] and self._nrows > 3 and self._ncols > 2: 
            table_data[-2] = [] * self._ncols

        
        # Finally build the table instance
        space_before, space_after = self._gen_table_space()
        table = Table(table_data,
                      style = table_style,
                      spaceBefore = space_before,
                      spaceAfter = space_after)

        #print(table.__dict__)
        return table
    
    def _gen_random_cn_sentence(self, length = [2, 6]):
        word_len = random_integer_from_list(length)
        #print('****************', word_len)
        #print('****************', self.cn_char)
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

    def _gen_random_cn_punctuation(self, length = [3, 8]):
        cn_leading = list('一二三四五六七八九十')
        sentence = self._gen_random_cn_sentence(length)
        return '、'.join([np.random.choice(cn_leading), sentence])
                                   
    def _decorate_underline(self, source_list, prob, table_style):
        """
        To add underline, the cell content has to be wrapped by Paragraph obj 
        To keep the same fome style with  the table content, one has to extract 
        font name/size from the table style. 
        """
        cell_font = [x[-1] for x in table_style._cmds if x[0] == 'FONT'][0]
        cell_font_size = [x[-1] for x in table_style._cmds if x[0] == 'FONTSIZE'][0]
        #align = [x[-1] for x in table_style._cmds if x[0] == 'ALIGNMENT'][0]
        cell_style = ParagraphStyle('temp', fontName = cell_font, fontSize = cell_font_size, alignment = 1)
        dst_list = [Paragraph('<u>' + s + '</u>', cell_style) if np.random.random() <= prob else s for s in source_list]
        return dst_list


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
        is_punct = False
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
            elif 'punctuation' in category:
                res = self._gen_random_cn_punctuation(self.config['content']['cn_chars']['word_length'])
                is_punct = True
            else:
                raise ValueError("category not recognized: %s"%category)

            # wrap text by probility 
            if 'wrap' in sub_block_config:
                if len(res) > 4 and np.random.random() < sub_block_config['wrap']:
                    mid = len(res) // 2 
                    res = '\n'.join([res[:mid], res[mid:]])
            result.append(res)

        if is_punct:
            result = [x if '、' in x else '  ' + x for x in result]
        return result

    def _gen_table_space(self):
        space_before = random_integer_from_list(self.config['layout']['space_before'])
        space_after = random_integer_from_list(self.config['layout']['space_after'])
        return space_before, space_after
