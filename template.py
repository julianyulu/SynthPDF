import os 
import numpy as np
from PIL import Image 
from reportlab.platypus import SimpleDocTemplate, Paragraph
from table import Table
from utils import random_integer_from_list

class myTemplate(SimpleDocTemplate):
    def __init__(self, filename, config, **kw):
        super().__init__(filename, **kw)
        self.config = config
        self.coords = []

    # def beforeFlowable(self, flowable):
    #     self.canv.rotate(90)
    #     print(self.frame)
    
    def afterFlowable(self, flowable):
        x_lowerLeft, y_lowerLeft = self.frame._x, self.frame._y
        x_upperRight, y_upperRight = self.frame._x2, self.frame._y2
        page_number = self.canv._pageNumber  # start from 1
        #print(type(flowable))
        #print(self.canv._pageNumber)
        #print(flowable.__dict__)
        #print(self.canv.__dict__)
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
        if hasattr(flowable, '_linecmds') and flowable._linecmds:
            cache_coords = []
            if np.random.random() < self.config['stamp']['prob']:            
                n_stamps = random_integer_from_list(self.config['stamp']['n'])
                # Add  n_stamps 
                for _ in range(n_stamps):
                    info = self._add_imgs('stamp', x_lowerLeft, y_lowerLeft,
                                          width, height, cache_coords)
                    info['page'] = page_number
                    self.coords.append(info)

            if np.random.random() < self.config['signature']['prob']:            
                n_signatures = random_integer_from_list(self.config['signature']['n'])
                # Add  n_signatures 
                for _ in range(n_signatures):
                    info = self._add_imgs('signature', x_lowerLeft, y_lowerLeft,
                                          width, height, cache_coords)
                    info['page'] = page_number
                    self.coords.append(info)


    def _is_overlap(self, new_stamp, cached_coords):
        """Check if new stamp overlaps with existing stamp coords 
        """
        x1, y1 = new_stamp[0]
        x2, y2 = new_stamp[1]
        for old_stamp in cached_coords:
            xmin, ymin = old_stamp[0]
            xmax, ymax = old_stamp[1]
            if (xmin < x1 < xmax or xmin < x2 < xmax) and (ymin < y1 < ymax or ymin < y2 < ymax):
                return True 
        return False

    def _add_imgs(self, kind, x, y, w, h, exist_coords):
        """
        x, y: coords of the table lower left corner, origins from lower left of the page
        Avoid stamp overlap 
        """
        cfg_object = self.config[kind]
        x_shift = random_integer_from_list(cfg_object['corner_dx'])
        y_shift = random_integer_from_list(cfg_object['corner_dy'])
        image_path = cfg_object['img_path']
        image_width = random_integer_from_list(cfg_object['width'])
        image_file = np.random.choice(os.listdir(image_path))

        # get original image size of image 
        image_img = Image.open(os.path.join(image_path, image_file))
        w_image, h_image = image_img.size

        # rescale to setted width (keepAspect = True in draw)
        h_image = image_width / w_image  * h_image 
        w_image = image_width
        
        max_iter = 50
        n_iter = 0
        while n_iter < max_iter:
            # choose one of the 4 corners and add shift
            if np.random.random() < 0.5: # image along top/bottom of table
                # Fix y in narrow range while having x range 
                y_image = int(np.random.choice([int(y), int(y + h + 1)]) - image_width // 2 + y_shift)
                x_image = np.random.randint(int(x), int(x + w)) - image_width //2  + x_shift
            else: #image along left / right of table 
                x_image = np.random.choice([int(x), int(x + w)]) - image_width // 2 + x_shift
                y_image = np.random.randint(int(y), int(y + h + 1)) - image_width //2  + y_shift
                
            image_coord = ((x_image, y_image),
                            (x_image + w_image, y_image + h_image))
            
            if self._is_overlap(image_coord, exist_coords):
                n_iter += 1
                continue
            else:
                exist_coords.append(image_coord)
                break
            
        img_width, img_height = self.canv.drawImage(os.path.join(image_path, image_file),
                                                    x_image, y_image, mask = 'auto',
                                                    anchor = 'sw', # anchored at center 
                                                    width = image_width,
                                                    preserveAspectRatio = True) 
                
        return {'kind': kind,
                'is_flowable': False, # page number added by the caller func
                'x': x_image,
                'y': y_image,
                'w': image_width,
                'h': int(img_height * image_width / img_width)}
