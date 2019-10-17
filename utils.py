import yaml
import os
import re
import base64
import codecs
from pdf2image import convert_from_path

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as fid:
        data = yaml.safe_load(fid)
    return data

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

def pdf2img(pdf_path, dpi = 150, image_path = 'output_imgs'):
    ## Below comment works for multi-page pdf
    if not os.path.exists(image_path): os.mkdir(image_path)
    pages = convert_from_path(pdf_path, dpi)
    file_prefix = os.path.basename(pdf_path).split('.pdf')[0]
    save_names = []
    for i,page in enumerate(pages):
        filename = file_prefix + '_page%d'%(i + 1) + '.jpg'
        save_name = os.path.join(image_path, filename)
        page.save(save_name, 'JPEG')
        save_names.append(save_name)
    return save_names

def b64pdf2img(b64_str, image_path, pdf_temp_path = 'pdf_files', dpi = 150):
    temp_pdf = b642pdf(b64_str, save_path = pdf_temp_path)
    img_save_path = pdf2img(temp_pdf, dpi = dpi, image_path = image_path)
    return img_save_path


test_file = 'config.yaml'
print(load_yaml(test_file)) 