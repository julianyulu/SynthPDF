import cv2
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.platypus import Spacer
from reportlab.lib.pagesizes import A4, landscape 
from elements.template import myTemplate
from elements.list import SynthList
from elements.header import PageHeader
from elements.table import SynthTable
from elements.title import SynthTitle, SynthSubTitle
from elements.paragraph import SynthParagraph
from elements.utils import random_integer_from_list, normalize_probabilty, pdf2img

class SynthPage:
    def __init__(self, config, filename = 'test.pdf'):
        self.filename = filename
        self.config = config
        self._initialize()
        self.doc = myTemplate(self._pdf_file,
                              config['notFlowables'],
                              pagesize = self._random_page_size(), bottomup = 1,
                              showBoundary = 0)
        
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

    def _random_page_size(self):
        cfg_page = self.config['page']

        prob_portrait = cfg_page['prob_portrait']
        prob_landscape = cfg_page['prob_landscape']

        prob_portrait, prob_landscape = normalize_probabilty([prob_portrait,prob_landscape])
        rand = np.random.random()
        if rand < prob_portrait:
            return A4
        else:
            return landscape(A4)
        
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

    def add_subtitle(self):
        subtitle = SynthSubTitle(self.config['subtitle']).subtitle
        self.elements.append(subtitle)

    def add_list(self):
        bullet_list = SynthList(self.config['list']).bullet_list
        self.elements.append(bullet_list)
        
    def as_pdf(self):
        self.doc.build(self.elements,
                       onFirstPage = PageHeader(self.config['page']['header']))
                       

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
        W_page, H_page = self.doc.pagesize
        factor = H / H_page 
        return (int(x * factor), int(H - y * factor))

    
    def annotate_box(self, show_img = False, save_img = False, save_json = False):
        ANNOTATE_ELEMENTS = ['table', 'text', 'stamp', 'signature']
        ELEMENTS_COLOR = {'table': (0, 0, 255),
                          'text': (255, 0, 0),
                          'stamp': (0, 255, 0),
                          'signature': (112, 22, 120)}
        FONT_FACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
        FONT_SCALE = 1
        FONT_THICKNESS = 1

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

            # Ensure coords are within the page range (for large tables)
            x_upperLeft = max(x_upperLeft, 0)
            y_upperLeft = max(y_upperLeft, 0)
            x_lowerRight = min(x_lowerRight, W)
            y_lowerRight = min(y_lowerRight, H)
        
            if elem['page'] == 1:
                kind = elem['kind']
                labels[idx] = {'kind': kind,
                               'p1': (int(x_upperLeft), int(y_upperLeft)),
                               'p2': (int(x_lowerRight), int(y_lowerRight))}
                idx += 1 
                if show_img or save_img:
                    if kind in ANNOTATE_ELEMENTS:
                        # bbox 
                        cv2.rectangle(img,
                                      (x_upperLeft, y_upperLeft),
                                      (x_lowerRight, y_lowerRight),
                                      ELEMENTS_COLOR[kind], 2)
                        retval, baseline = cv2.getTextSize(kind, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
                        # text backgound 
                        width, height = retval
                        cv2.rectangle(img,
                                      (x_upperLeft, y_upperLeft - height),
                                      (x_upperLeft + width, y_upperLeft),
                                      (255, 255, 204), -1) # cany text backgroud
                        # text
                        text = cv2.putText(img, kind, (x_upperLeft, y_upperLeft),
                                           FONT_FACE, FONT_SCALE,
                                           ELEMENTS_COLOR[kind], FONT_THICKNESS)
                        
                        
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

    def annotate_mask(self, show_img = False, save_img = True):
        ANNOTATE_LABELS = {'table': 1, 'text': 2, 'stamp':3, 'signature':4}
        annot = self.doc.coords
        # current only annote the first page, as the coords has no info about the page
        img_files = self.as_img()
        img = cv2.imread(img_files[0])
        H, W = img.shape[:2]

        labels = {}
        idx = 0 
        prev_ycoords = -1  # filter out the annot on the second page

        mask = np.zeros([H, W])
        #mask = np.ones([H, W])  * 10 
        for elem in annot:
            if elem['page'] == 1:
                x_lowerLeft, y_lowerLeft = elem['x'], elem['y']
                w_elem, h_elem = elem['w'], elem['h']
                x_upperLeft, y_upperLeft = self._trans_coords((x_lowerLeft, y_lowerLeft + h_elem), W, H)
                x_lowerRight, y_lowerRight = self._trans_coords((x_lowerLeft + w_elem, y_lowerLeft), W, H)

                # Ensure coords are within the page range (for large tables)
                x_upperLeft = max(x_upperLeft, 0)
                y_upperLeft = max(y_upperLeft, 0)
                x_lowerRight = min(x_lowerRight, W)
                y_lowerRight = min(y_lowerRight, H)

                print("*", elem['kind'], "*")
                if elem['kind'] in ANNOTATE_LABELS:
                    print(elem['kind'], ANNOTATE_LABELS[elem['kind']])
                    mask[int(y_upperLeft) : int(y_lowerRight), int(x_upperLeft):int(x_lowerRight)] = ANNOTATE_LABELS[elem['kind']]
                
        if save_img:
            # note: mask has to be 'png' format, 'jpg' will change the annotations 
            save_name = os.path.basename(img_files[0]).split('_page1.jpg')[0] + '_mask.png' 
            cv2.imwrite(os.path.join(self._img_path, save_name), mask)
        if show_img:
            plt.figure(figsize = (12, 20))
            plt.imshow(mask*255, 'gray')
            plt.show()
