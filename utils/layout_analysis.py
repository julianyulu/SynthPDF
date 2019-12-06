from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LTTextBoxHorizontal


document = open('test.pdf', 'rb') 
#Create resource manager
rsrcmgr = PDFResourceManager()
# Set parameters for analysis.
laparams = LAParams(char_margin = 1.0)
# Create a PDF page aggregator object.
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)
for page in PDFPage.get_pages(document):
    interpreter.process_page(page)
    # receive the LTPage object for the page.
    layout = device.get_result()
    for element in layout:
        print(element)
        #if instanceof(element, LTTextBoxHorizontal):
        #    print(element.get_text())
