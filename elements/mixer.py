import numpy as np 
from elements.page import SynthPage

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
            prob_book[key] = [acc_prob, acc_prob + prob_book[key] / sum_prob]
            acc_prob = prob_book[key][1]
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

        # then double check if avoid neighbor table
        if self.config['mixer']['avoid_neighbor_tables']:
            prev = ''
            for i in range(len(select_elements)):
                if '_table' in select_elements[i] and '_table' in prev:
                    select_elements[i] = 'add_paragraph'
                prev = select_elements[i]
                
        # add elements to page from select_elements 
        for op in select_elements:
            self.page.__getattribute__(op)()

        # when there is not interested element, skip 
        if len(self.page.doc.coords) == 0: return
        
        # finally output result file 
        if self.config['mixer']['as_pdf']:
            self.page.as_pdf()

        if self.config['mixer']['as_img']:
            self.page.as_img()

        if self.config['mixer']['annotate_box']:
            annot = self.page.annotate_box(save_img = self.config['mixer']['save_annotate_imgs'],
                                       show_img = self.config['mixer']['show_annotate_imgs'],
                                       save_json = self.config['mixer']['save_single_annotate_json'])
        if self.config['mixer']['annotate_mask']:
            annot = self.page.annotate_mask(save_img = self.config['mixer']['save_annotate_imgs'],
                                            show_img = self.config['mixer']['show_annotate_imgs'])
            return annot 
