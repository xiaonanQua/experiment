import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import cv2
from keras.models import load_model
import time

# In[2]:


model_smoke_detector = load_model('model/smoke_detecctor_with_mobilenet.h5')

# In[3]:cap = cv2.VideoCapture('/home/team/sf6_1.avi')


model_mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# In[4]:


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('E:/smoke_video/SF6_1.avi')
# video_writer = cv2.VideoWriter('smoke_detection_output1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
background = None

out = cv2.VideoWriter('save.avi',
                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))

steps = [5, 150, 500]
# frames = [0] * 30
choice = 0
total_time = 0
count = 0
# num = 1
while True:
    #status, img = cap.read()
    # 读取视频流
    grabbed, frame_lwpCV = cap.read()
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_lwpCV
        continue
    # 对于每个从背景之后读取的帧都会计算其与北京之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
    diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀

    # 显示矩形框
    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) < 1500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        # 按'q'健退出循环
        if key == ord('q'):
            break
    # img = img[:, ::-1, :]
    #if status:
        # img = img[100: 500, 200: 600, :]
        '''img_detect = cv2.resize(img[:, :, [2, 1, 0]], (48 * num, 48 * num))
        detection_results = []
        for i in range(num):
            for j in range(num):
                tmp = img_detect[i * 48: (i + 1) * 48, j * 48: (j + 1) * 48, :]
                tmp = cv2.resize(tmp, (224, 224))
                start_time = time.time()
                x = image.img_to_array(tmp)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = model_mobilenet.predict(x).reshape((1, 7*7*1024))
                #print(model_smoke_detector.predict(feature)[0, 0])
                result = model_smoke_detector.predict(feature)[0, 0]
                detection_results.append(result)
                end_time = time.time()
                total_time += end_time - start_time
                count += 1'''

        img_detect = cv2.resize(image[:, :, [2, 1, 0]], (224, 224))
        # tmp = img_detect[i * 48: (i + 1) * 48, j * 48: (j + 1) * 48, :]
        # tmp = cv2.resize(tmp, (224, 224))
        start_time = time.time()
        x = image.img_to_array(img_detect)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model_mobilenet.predict(x).reshape((1, 7 * 7 * 1024))
        # print(model_smoke_detector.predict(feature)[0, 0])
        result = model_smoke_detector.predict(feature)[0, 0]
        # detection_results.append(result)
        end_time = time.time()

        total_time += end_time - start_time
        count += 1

        # frames.pop(0)
        # frames.append(1 if result > 0.5 else 0)
        # img = cv2.flip(img, 1)
        cv2.putText(image,
                    'Probability: {0}'.format(str(result)),
                    (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    1)
        cv2.putText(image,
                    '{0}'.format('smoke' if result > 0.5 else 'no smoke'),
                    (30, 90),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    1)
        '''cv2.putText(img,
                    '{0}'.format('smoke' if sum(frames) > 0 else 'no smoke'),
                    (30, 90), 
                    cv2.FONT_HERSHEY_COMPLEX, 
                    2, 
                    (0, 0, 255), 
                    2)'''
        out.write(image)
        cv2.imshow('', image)
        key = cv2.waitKey(steps[choice])
        # video_writer.write(img)
        if key == 27:
            break
        elif key == ord('n'):
            choice += 1
            choice %= 3
    else:
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()
# video_writer.release()


# In[5]:


#total_time / count
