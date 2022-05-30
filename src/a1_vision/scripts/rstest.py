#!/usr/bin/python3
'''
by yzh 2022.2.13
'''
# 导入依赖
import random
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch

import pyrealsense2 as rs
import math
import yaml
import argparse
import os
import time
import numpy as np
import sys
import rospy
from std_msgs.msg import Float64MultiArray
from itertools import chain

import cv2
# PyTorch
# YoloV5-PyTorch
print(rs.config())
pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
def generate_matrix(theta):
    theta = theta/180*np.pi
    R_1 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])@ R_1
    # p = (np.array([0.29,0,0.055])+np.array([0.0325*np.cos(theta),-0.0075,-0.0325*np.sin(theta)])).T
    p = (np.array([0.295,0,0.055+0.176])+np.array([0.0420*np.cos(theta),0.034,-0.0420*np.sin(theta)])).T  
    # print('R',R)
    # print('p:',p)
    return R,p
    
class YoloV5:
    def __init__(self, yolov5_yaml_path='config/yolov5s.yaml'):
        '''初始化'''
        # 载入配置文件
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # 随机生成每个类别的颜色
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for class_id in range(self.yolov5['class_num'])]
        # 模型初始化
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        '''模型初始化'''
        # 设置日志输出
        set_logging()
        # 选择计算设备
        device = select_device(self.yolov5['device'])
        # 如果是GPU则使用半精度浮点数 F16
        is_half = device.type != 'cpu'
        # 载入模型
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  # 载入全精度浮点数的模型
        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())  # 检查模型的尺寸
        if is_half:
            model.half()  # 将模型转换为半精度
        # 设置BenchMark，加速固定图像的尺寸的推理
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 图像缓冲区初始化
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
        # 创建模型
        # run once
        _ = model(img_torch.half()
                  if is_half else img) if device.type != 'cpu' else None
        self.is_half = is_half  # 是否开启半精度
        self.device = device  # 计算设备
        self.model = model  # Yolov5模型
        self.img_torch = img_torch  # 图像缓冲区

    def preprocessing(self, img):
        '''图像预处理'''
        # 图像缩放
        # 注: auto一定要设置为False -> 图像的宽高不同
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        # print("img resize shape: {}".format(img_resize.shape))
        # 增加一个维度
        img_arr = np.stack([img_resize], 0)
        # 图像转换 (Convert) BGR格式转换为RGB
        # 转换为 bs x 3 x 416 x
        # 0(图像i), 1(row行), 2(列), 3(RGB三通道)
        # ---> 0, 3, 1, 2
        # BGR to RGB, to bsx3x416x416
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # 数值归一化
        # img_arr =  img_arr.astype(np.float32) / 255.0
        # 将数组在内存的存放地址变成连续的(一维)， 行优先
        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        # https://zhuanlan.zhihu.com/p/59767914
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        '''模型预测'''
        # 图像预处理
        img_resize = self.preprocessing(img)  # 图像缩放
        self.img_torch = torch.from_numpy(img_resize).to(self.device)  # 图像格式转换
        self.img_torch = self.img_torch.half(
        ) if self.is_half else self.img_torch.float()  # 格式转换 uint8-> 浮点数
        self.img_torch /= 255.0  # 图像归一化
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        # 模型推理
        t1 = time_sync()
        pred = self.model(self.img_torch, augment=False)[0]
        # pred = self.model_trt(self.img_torch, augment=False)[0]
        # NMS 非极大值抑制
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        t2 = time_sync()
        # print("推理时间: inference period = {}".format(t2 - t1))
        # 获取检测结果
        det = pred[0]
        gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]]  # [w, h, w, h]

        if view_img and canvas is None:
            canvas = np.copy(img)
        xyxy_list = []
        conf_list = []
        class_id_list = []
        c1_list = []
        c2_list = []
        if det is not None and len(det):
            # 画面中存在目标对象
            # 将坐标信息恢复到原始图像的尺寸
            det[:, :4] = scale_coords(
                img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                class_id_list.append(class_id)
                if view_img:
                    # 绘制矩形框与标签
                    label = '%s %.2f' % (
                        self.yolov5['class_name'][class_id], conf)
                    self.plot_one_box(
                         xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
                    c1,c2 = self.getc1c2(
                         xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
                    c1_list.append(c1)
                    c2_list.append(c2)
        return canvas, class_id_list, xyxy_list, conf_list,c1_list,c2_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''绘制矩形框+标签'''
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    def getc1c2(self, x, img, color=None, label=None, line_thickness=None):
        ''''绘制矩形框+标签'''
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        return c1, c2
        

if __name__ == '__main__':

    rospy.init_node('vision',anonymous=True)
    pospub = rospy.Publisher('/meihuazhuang_pos',Float64MultiArray,queue_size=10)
    print("[INFO] YoloV5目标检测-程序启动")
    print("[INFO] 开始YoloV5模型加载")
    # YOLOV5模型配置文件(YAML格式)的路径 yolov5_yaml_path
    model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
    print("[INFO] 完成YoloV5模型加载")
    # pub = rospy.Publisher("/test_camera", Float64MultiArray, queue_size=10)
    # rospy.init_node('talker', anonymous=True)
    # qx = 1*np.pi / 180  
    # qy = 1*np.pi / 180
    # qz = 1*np.pi / 180
    # rx = np.array([[1, 0, 0], [0, np.cos(qx), -np.sin(qx)], [0, np.sin(qx), np.cos(qx)]])
    # ry = np.array([[np.cos(qy), 0, np.sin(qy)], [0, 1, 0], [-np.sin(qy), 0, np.cos(qy)]])
    # rz = np.array([[np.cos(qz), -np.sin(qz), 0], [np.sin(qz), np.cos(qz), 0], [0, 0, 1]])
    # r = np.matmul(rx, np.matmul(ry, rz) ) #rotation matrix
    # T = np.array(r)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
            if not depth_image.any() or not color_image.any():
                continue
            # Convert images to numpy arrays
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images

            t_start = time.time()  # 开始计时
            # YoloV5 目标检测
            canvas, class_id_list, xyxy_list, conf_list,c1_list,c2_list = model.detect(
                color_image)

            t_end = time.time()  # 结束计时\
            #canvas = np.hstack((canvas, depth_colormap))
            #print(class_id_list)

            camera_xyz_list= []
            # print("shape", len(xyxy_list), c1_list, c2_list)
            if xyxy_list:
                for i in range(len(xyxy_list)):
                    ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)  # 计算像素坐标系的x
                    uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)  # 计算像素坐标系的y
                    dis = aligned_depth_frame.get_distance(ux, uy)
                    #camera_xyz = rs.rs2_deproject_pixel_to_point(
                    #    depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                    #camera_xyz = np.matmul(r, camera_xyz)
                    camera_pos = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, (ux, uy), dis)  # 计算相机坐标系的xyz
                    camera_pos = np.round(np.array(camera_pos), 5)  # 转成5位小数
                    R,p = generate_matrix(90)   # camera rotation angle
                    # print('-----',np.matmul(R,camera_pos))
                    camera_xyz = R @ camera_pos + p
                    camera_xyz = np.round(np.array(camera_xyz), 5)  # 转成5位小数
                    camera_xyz = camera_xyz.tolist()

                    cv2.circle(canvas, (ux,uy), 4, (255, 255, 255), 5)#标出中心点

                    
                    # print(c1_list[i])
                    # computer rectangular in real
                    c1x = int(c1_list[i][0])  # 计算像素坐标系的x
                    c1y = int(c1_list[i][1])  # 计算像素坐标系的y
                    c1_pos = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, (c1x, c1y), dis)  # 计算相机坐标系的xyz
                    c1_pos = np.round(np.array(c1_pos), 5)  # 转成5位小数
                    #print('camera:',camera_pos)
                    #R, p = generate_matrix(90)
                    # print('-----',np.matmul(R,camera_pos))
                    c1_xyz = R @ c1_pos + p
                    c1_xyz = np.round(np.array(c1_xyz), 5)  # 转成5位小数
                    c1_xyz = c1_xyz.tolist()

                    # computer rectangular in real
                    c2x = int(c2_list[i][0])  # 计算像素坐标系的x
                    c2y = int(c2_list[i][1])  # 计算像素坐标系的y
                    c2_pos = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, (c2x, c2y), dis)  # 计算相机坐标系的xyz
                    c2_pos = np.round(np.array(c2_pos), 5)  # 转成5位小数
                    #print('camera:',camera_pos)
                    #R, p = generate_matrix(90)
                    # print('-----',np.matmul(R,camera_pos))
                    c2_xyz = R @ c2_pos + p
                    c2_xyz = np.round(np.array(c2_xyz), 5)  # 转成5位小数
                    c2_xyz = c2_xyz.tolist()
                    # print(c1_xyz, c2_xyz)
                    detected_diameter_size = np.linalg.norm(np.array(c1_xyz) - np.array(c2_xyz))/ (2 ** 0.5)
                    cv2.putText(canvas, str(round(detected_diameter_size, 5)), (ux+20, uy+10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#标出坐标
                    if (detected_diameter_size > 0.145):
                        print('Complete target!---------add')
                        print(str(detected_diameter_size), str(camera_xyz))
                        camera_xyz.append(detected_diameter_size)
                        camera_xyz_list.append(camera_xyz)

            camera_flatten_list = list(chain.from_iterable(camera_xyz_list))
            

            # print(camera_flatten_list)
            meihuazhuangpos = Float64MultiArray(data = camera_flatten_list)
            # print(meihuazhuangpos.data)
            pospub.publish(meihuazhuangpos)

            # 添加fps显示
            fps = int(1.0 / (t_end - t_start))
            # print(fps)
            #cv2.putText(c                lineType=cv2.LINE_AA, color=(0, 0, 0))
            # cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
            #                cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('detection', canvas)
            key = cv2.waitKey(1)
            # # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
