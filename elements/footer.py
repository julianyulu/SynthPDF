from reportlab.pdfgen import canvas

class PageFooter:
    def __init__(self, config):
        self.config = config

    def __call__(self, canvas, doc):
        canvas.saveState()
        w, h = canvas._pagesize
        font_size = random_integer_from_list(self.config['text']['font_size'])
        line_y = random_integer_from_list(self.config['text']['bottom_margin'])
        canvas.setFontSize(font_size)
        canvas.drawCentredString(w/2, line_y, str(canvas.getPageNumber()))
        canvas.restoreState()
