import yaml
import os
import re
import base64
import codecs
import numpy as np

from pdf2image import convert_from_path

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as fid:
        data = yaml.safe_load(fid)
    return data

def normalize_probabilty(prob_list):
    tot = sum(prob_list)
    probs = [x / tot for x in prob_list]
    return probs 

def random_integer_from_list(inlist, seed = None, empty_default = None):
    if seed is None:
        seed = np.random.seed()
    if empty_default is not None and len(inlist) == 0: return empty_default
    elif len(inlist) == 1: return inlist[0]
    elif len(inlist) == 2: return np.random.randint(inlist[0], inlist[1] + 1)
    else: return np.random.choice(inlist)

def prob2category(probdict):
        items = [x for x in probdict.keys() if x.startswith('prob_')]
        probs = [probdict[x] for x in items]
        if not sum(probs) == 1:
            tot = sum(probs)
            probs = [x / tot for x in probs]
        acc = 0
        bound_dict ={}
        for p, item in zip(probs, items):
            bound_dict[','.join([str(acc), str(p + acc)])] = item
            acc += p
            
        def prob2item(x):
            for key in bound_dict:
                lb, ub = key.split(',')
                lb, ub = float(lb), float(ub)
                if lb <= x <= ub:
                    return bound_dict[key]
            raise ValueError("prob2item found not item from dictionary ", bound_dict)
        return prob2item
    
def pdf2b64(pdf_file):
    with open(pdf_file, 'rb') as pf:
        encoded_str = base64.b64encode(pf.read())
    return encoded_str.decode('utf-8')
    
def b642pdf(b64str, save_path = 'pdf_files'):
    if not os.path.exists(save_path): os.mkdir(save_path)
    #filename = re.sub('\D', "", b64str[-10:])
    filename = ''.join(re.findall('\w+', b64str[-10:])) + '.pdf'
    save_name = os.path.join(save_path, filename)
    with open(save_name, 'wb') as pf:
        pf.write(base64.b64decode(b64str)) # encode here converts str to bytes
    return save_name 

def pdf2img(pdf_file, save_path = 'output_imgs', dpi = 150):
    ## Below comment works for multi-page pdf
    if not os.path.exists(save_path): os.mkdir(save_path)
    pages = convert_from_path(pdf_file, dpi)
    file_prefix = os.path.basename(pdf_file).split('.pdf')[0]
    save_names = []
    for i,page in enumerate(pages):
        filename = file_prefix + '_page%d'%(i + 1) + '.jpg'
        save_name = os.path.join(save_path, filename)
        page.save(save_name, 'JPEG')
        save_names.append(save_name)
    return save_names

def b64pdf2img(b64_str, image_path, pdf_temp_path = 'pdf_files', dpi = 150):
    temp_pdf = b642pdf(b64_str, save_path = pdf_temp_path)
    img_save_path = pdf2img(temp_pdf, dpi = dpi, image_path = image_path)
    return img_save_path

### Test load_yaml
# test_file = 'config.yaml'
# print(load_yaml(test_file)) 

### Test prob2category
# test_dict = {'a': 0.2, 'b':0.5, 'c': 0.3}
# func = prob2category(test_dict)
# print(func(np.random.random()))


