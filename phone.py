import cv2
 
cv2.namedWindow("camera", 1)
# 开启ip摄像头
video = "http://admin:admin@192.168.0.48/8081"  # 此处@后的ipv4 地址需要改为app提供的地址
cap = cv2.VideoCapture(video)
#　　开摄像头
while True:
    # Start Camera, while true, camera will runq
 
    ret, image_np = cap.read()
 
    # Set height and width of webcamq
    height = 600
    width = 1000
 
    # Set camera resolution and create a break function by pressing 'q'
    cv2.imshow('object detection',image_np)# cv2.resize(image_np, (width, height)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    print(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
# Clean up
cap.release()
cv2.destroyAllWindows()
 
#保存文件
 
fps = 30
 
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
 
# 调用VideoWrite（）函数
videoWrite = cv2.VideoWriter('MySaveVideo.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
 
# 先获取一帧，用来判断是否成功调用摄像头
success, frame = cap.read()
 
# 通过设置帧数来设置时间,减一是因为上面已经获取过一帧了
numFrameRemainling = fps * 10 - 1
 
# 通过循环保存帧
while success and numFrameRemainling > 0:
    videoWrite.write(frame)
    success, frame = cap.read()
    numFrameRemainling -= 1
 
# 释放摄像头
cap.release()
