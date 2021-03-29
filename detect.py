import argparse
import time,os,sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.parameters import opt, a_path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#获取参数信息
opt = opt() 
#opt.list_all_member()


def detect(save_img=True):
    # 获取输出文件夹，输入源，权重，参数等参数
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    # 获取设备  
    device = select_device(opt.device)
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    # 设置数据加载方式
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # 获取类别名字
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色
    colors = [[random.randint(0, 1) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # 进行一次前向推理,测试程序是否正常
    t0 = time.time()
   
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # 对每一张图片作处理
        for i, det in enumerate(pred):  # detections per image
            # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 设置保存图片/视频的路径
        
            save_path = a_path.images_path+str(p.name)  # img.jpg
            
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            aa = 0
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 在原图上画框
                        aa = 1
                        #print(xywh)
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        sp = im0.shape
                        #获取坐标
                        left = int((float(xywh[0]) * sp[1]) - (float(xywh[2]) * sp[1]/2))
                        upper = int((float(xywh[1]) * sp[0]) - (float(xywh[3]) * sp[0]/2))
                        right = int((float(xywh[0]) * sp[1]) + (float(xywh[2]) * sp[1]/2))
                        lower = int((float(xywh[1]) * sp[0]) + (float(xywh[3]) * sp[0]/2))
                        #裁剪
                        im0 = im0[upper:lower, left:right]

            # Stream results
            # 如果设置展示，则show图片/视频
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 设置保存图片/视频 有label则保存
            vid_path, vid_writer = None, None
            try:
                if save_img & aa == 1:
                #if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print("Could not convert data to an integer.")
            except IOError:
                print ("Error: 没有找到文件或读取文件失败")
            except:
                print("Unexpected error:", sys.exc_info()[0])
