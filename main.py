import uuid
from elements.utils import load_yaml
from elements.page import SynthPage
from elements.mixer import PageMixer
from tqdm import tqdm
from multiprocessing import Pool

# ======================= Generate using config file  ============================

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

# ======================= Generate using customized elements  ============================

# if __name__ == '__main__':    
#     config =  load_yaml('config_large_table.yaml')
#     cfg_runner = config['runner']
#     run_parallel = cfg_runner['run_parallel']
#     n_files = cfg_runner['n_files']
#     n_processors = cfg_runner['num_processors']
#     def runner(filename):
#         sp = SynthPage(config, filename = filename)
#         sp.add_spacer()
#         sp.add_table()
#         sp.add_spacer()
#         sp.as_pdf()
#         sp.as_img()
#         #sp.annotate_box(save_img = True)
#         sp.annotate_mask(save_img = True)        
#     if run_parallel:
#         filenames = [uuid.uuid4().hex + '.pdf' for _ in range(n_files)]
#         SynthPage(config)._initialize() # make output folders to avoid parallel conflict 
#         with Pool(n_processors) as p:
#             _ = list(tqdm(p.imap_unordered(runner, filenames), total = n_files))
#     else:
#         for _ in tqdm(range(n_files)):
#             filename = uuid.uuid4().hex + '.pdf'
#             runner(filename)

# ======================= test run ============================

# from elements.table import SynthTable
# from elements.paragraph import SynthParagraph
# #from elements.title import SynthTitle
# config =  load_yaml('config.yaml')

# # #Test Table 
# # tb = SynthTable(config['table'])
# # print(tb.nrows, tb.ncols)
# # print(tb.table)

# # #Test Paragraph
# # pa = SynthParagraph(config['paragraph'])
# # print(pa.paragraph)

# ##Test SynthPage
# sp = SynthPage(config)
# #sp.add_paragraph()
# sp.add_title()
# #sp.add_spacer()
# sp.add_list()
# #sp.add_table()
# sp.add_paragraph()
# sp.add_table()
# #sp.add_title()
# sp.as_pdf()
# sp.as_img()
# sp.annotate_box(save_img = True)
# sp.annotate_mask(save_img = True)

#Test Mixer 
#m = PageMixer(config)
#m.make()
#print(m.page.doc.coords)
