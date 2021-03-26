from flask import Flask
from detect import *
from detect_s import *
from bianhuan import *
from main import l_clear, shu
from utils.parameters import get_url_picture

app = Flask(__name__)

@app.route('/<path:path>')
def hello_world(path):
    #print(path)
    #path='https://img14.360buyimg.com/n0/jfs/t1/159901/3/11391/351369/6046d0d9E8e2127d6/e2a03621f7bfe40b.jpg'
    #每次运行前清空exp文件夹
    l_clear()

    get_url_picture(path)
    #定位、切割
    detect()
    #图像变换
    bianhuan()
    #识别
    detect_s()
    #输出price
    x = shu()
    return x

if __name__ == '__main__':
   app.run(debug = True)