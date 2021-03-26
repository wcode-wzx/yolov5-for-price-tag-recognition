from detect import detect
from detect_s import *
from bianhuan import *
from utils.parameters import get_url_picture, url
import shutil 


mm = []
price = []

class a_path():
    images_path = 'runs/detect/exp/images/'
    cut_path = ''
    labels_path = 'runs/detect/exp/labels/'
    price_path = 'runs/detect/exp/price/'

def l_clear():
    shutil.rmtree('runs\detect\exp')  
    os.mkdir('runs\detect\exp') 
    os.mkdir(a_path.images_path) 
    os.mkdir(a_path.labels_path) 
    os.mkdir(a_path.price_path) 

def rr_name():
    rname = os.listdir(a_path.labels_path) 
    return rname

def shu():
    rrname = rr_name()
    for l in rrname:
        #print(l)
        # Open file   
        fileHandler  =  open  (a_path.labels_path + str(l),  "r")
        # Get list of all lines in file
        listOfLines  =  fileHandler.readlines()
        # Close file
        fileHandler.close()
        for  line in  listOfLines:
            mm.append([line.strip()[2:7],line.strip()[0]])
            mm.sort()
        
        for ii in range(0,len(mm)):
            x  = mm[ii][1]
            price.append(x)
        y = ''.join(price)
        #print(mm,*price)
        mm.clear()
        price.clear()
        with open(a_path.price_path+str(l).split('.')[0]+'.'+str(y)+".txt","w") as f:
                f.write(y)
        if url:
            pass 
        else:
            with open("runs/detect/exp/price.txt","a") as f:
                y = '![yuantu](E:\\vsProject\YOLOv5\yolov5_detect\data\images\\'+str(l).split('.')[0]+'.jpg)'+'\n'+'\n'+'![price](E:\\vsProject\YOLOv5\yolov5_detect\\runs\detect\exp\images\\'+str(l).split('.')[0]+'.jpg)'+'\n'+'\n'+y+'\n'+'\n'
                f.write(y) 
     

if __name__ == '__main__':
    #是否读取url图片
    if url:
        img_src_list=['https://img14.360buyimg.com/n0/jfs/t1/159901/3/11391/351369/6046d0d9E8e2127d6/e2a03621f7bfe40b.jpg','https://img14.360buyimg.com/n0/jfs/t1/157421/22/15124/169358/60593593E03a6ea7d/7acf5f44b6f63df7.jpg']
        get_url_picture(img_src_list)
    else:
        pass
    #每次运行前清空exp文件夹
    l_clear()
    #定位、切割
    detect()
    #图像变换
    bianhuan()
    #识别
    detect_s()
    #输出price
    shu()

    print("************complete**************")