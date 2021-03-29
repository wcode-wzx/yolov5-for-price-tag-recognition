import shutil , os
from utils.parameters import a_path

mm = []
price = []

def cclear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def l_clear():
    cclear(a_path.get_path)
    cclear(a_path.images_path)
    cclear(a_path.labels_path)
    cclear(a_path.price_path)

def rr_name():
    rname = os.listdir(a_path.labels_path) 
    return rname

def shu():
    rrname = rr_name()
    for l in rrname:
        #print(l)
        # Open file   
        fileHandler  =  open(a_path.labels_path + str(l),  "r")
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