import urllib
import urllib.request

def get_url_picture(img_src):
    try:
        request = urllib.request.Request(img_src)
        response = urllib.request.urlopen(request)
        get_img = response.read()
        with open('data/cache_p/'+img_src.split('/')[-1],'wb') as fp:
            fp.write(get_img)
        print('图片加载完成')
    except:
        print('访问空')

class opt(object):
    def __init__(self):
        self.source = 'data/cache_p'
        self.agnostic_nms = False 
        self.augment = False
        self.classes = None 
        self.conf_thres=0.25 
        self.device='0'
        self.exist_ok=False
        self.img_size=640
        self.iou_thres=0.45
        self.name='exp'
        self.project='runs/detect'
        self.save_conf=False
        self.save_txt=True
        self.view_img=False
        self.weights='weights/dingwei.pt'
    def list_all_member(self):
        for name,value in vars(self).items():
            print('%s=%s'%(name,value))


class opt2(object):
    def __init__(self):
        self.source = 'runs/detect/exp/images/'
        self.agnostic_nms = False 
        self.augment = False
        self.classes = None 
        self.conf_thres=0.25 
        self.device='0'
        self.exist_ok=False
        self.img_size=640
        self.iou_thres=0.45
        self.name='exp'
        self.project='runs/detect'
        self.save_conf=False
        self.save_txt=True
        self.view_img=False
        self.weights='weights/shibie.pt'
    def list_all_member(self):
        for name,value in vars(self).items():
            print('%s=%s'%(name,value))
