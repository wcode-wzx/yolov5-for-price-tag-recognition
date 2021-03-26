import shutil , os

mm = []
price = []

class a_path():
    images_path = 'runs/detect/exp/images/'
    cut_path = ''
    labels_path = 'runs/detect/exp/labels/'
    price_path = 'runs/detect/exp/price/'

def l_clear():
    if os.path.exists('data/cache_p'):
        shutil.rmtree('data/cache_p')
        os.mkdir('data/cache_p')
    else:
        os.mkdir('data/cache_p') 
    shutil.rmtree('runs/detect/exp') 
    os.mkdir('runs/detect/exp') 
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
        return y