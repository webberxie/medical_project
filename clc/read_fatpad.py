#数据读取
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
#from skimage import morphology
import os

from math import *
import SimpleITK as sitk
from PIL import Image
import cv2
import random
#import xlrd
import pandas
import openpyxl

NII_DIR = '/media/chen/Backup Plus/datasets/knee/fat_pad/IPFP/train'  # nii文件所在root目录
NII_DIR_validate = '/media/chen/Backup Plus/datasets/knee/fat_pad/IPFP/validate'
NII_DIR_test = '/media/chen/Backup Plus/datasets/knee/fat_pad/IPFP/test'  # nii文件所在root目录

def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    '''
    image_sort = np.sort(image)
    MAX_BOUND = image_sort[-int(0.0001 * len(image))]
    MIN_BOUND = image_sort[int(0.0001 * len(image))]
    '''
    MAX_BOUND = image.max()
    MIN_BOUND = image.min()
    image_all = 255 * ( (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) )
    return image_all

def histeq(im, nbr_bins=256):
    """对一幅灰度图像进行直方图均衡化"""
    # 计算图像的直方图
    # 在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  #
    cdf = 255.0 * cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def rotate(image, degree, center=None, scale=1.0):
    height, width = image.shape[:2]
    heightNew = int(width * abs(np.sin(radians(degree))) + height * abs(np.cos(radians(degree))))
    widthNew = int(height * abs(np.sin(radians(degree))) + width * abs(np.cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1) #构造旋转矩阵，（旋转中心，角度，缩放比例）

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    img_rota = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0)) #进行仿射变换，（输入图像，输出图像，输出尺寸，边界取值）
    rows, cols = img_rota.shape  # 旋转后图像的行，列，通道
    # print('img_src.shape:', img_src.shape)
    max_len = max(rows, cols)
    img_bg = np.zeros((max_len, max_len, 1), np.uint8)
    img_bg.fill(0)  # 填充黑色
    # padding
    if rows > cols:
        len_padding = int((max_len - cols) / 2)
        if (max_len - 2 * len_padding) - cols > 0:
            img_bg[:, len_padding: -len_padding - 1, :] = img_rota
        elif (max_len - 2 * len_padding) - cols < 0:
            img_bg[:, len_padding: -len_padding + 1, :] = img_rota
        else:
            img_bg[:, len_padding: -len_padding, :] = img_rota

    elif rows < cols:
        len_padding = int((max_len - rows) / 2)
        if (max_len - 2 * len_padding) - rows > 0:
            img_bg[len_padding: -len_padding - 1, :, :] = img_rota
        elif (max_len - 2 * len_padding) - rows < 0:
            img_bg[len_padding: -len_padding + 1, :, :] = img_rota
        else:
            img_bg[len_padding: -len_padding, :, :] = img_rota
    else:
        img_bg = img_rota
    img_bg = cv2.resize(img_bg,(192,192))
    return img_bg

def rotate_3d(data,angle):
    depth, width, height = 16, 192, 192
    new_data = np.zeros(shape=(depth, width, height))
    for i in range(depth):
        new_data[i, :, :] = rotate(data[i, :, :],angle)
    return new_data


def flip(data,flip_type): #flip_type:0,1
    if flip_type == 1:
        flipcode = 1
    elif flip_type == 2:
        flipcode = -1
    elif flip_type == 3:
        flipcode = 0
    else:
        flipcode = 2
    if flipcode ==2:
        return data
    depth,width,height = 16,192,192
    new_data = np.zeros(shape=(depth,width,height))
    for i in range(depth):
        new_data[i,:,:] = cv2.flip(data[i,:,:],flipcode,dst=None)
    return new_data

def read_nii_file(img_path,random_crop_type):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = sitk.ReadImage(img_path)
    image_arr = sitk.GetArrayFromImage(nii_image)

    # 随机裁剪

    if random_crop_type == 0:
        image_arr_back = norm_img(image_arr[0, 150:342, 30:222])  # 裁剪到192*192
    elif random_crop_type == 1:
        image_arr_back = norm_img(image_arr[0, 130:322, 10:202])  # --
    elif random_crop_type == 2:
        image_arr_back = norm_img(image_arr[0, 170:362, 50:242])  # ++
    elif random_crop_type == 3:
        image_arr_back = norm_img(image_arr[0, 130:322, 50:242])  # -+
    else:
        image_arr_back = norm_img(image_arr[0, 170:362, 10:202])  # +-
    return image_arr_back

def read_label(file_path, file):
    nii_image = sitk.ReadImage(os.path.join(file_path, file))
    image_arr = sitk.GetArrayFromImage(nii_image)
    return image_arr[:, 150:374, 30:254]

def people_image_train(people_dir): #截取输入数据到（32，224，224）
    files_singles = os.listdir(people_dir)
    people_image = np.zeros(shape=(16, 192, 192))
    # 随机裁剪
    random_crop_type = random.randint(0, 4)
    for f in files_singles:
        image_2d = read_nii_file(os.path.join(people_dir, f),random_crop_type) # 读取图片
        if int(f) >= 10 and int(f) < 26:
            people_image[int(f)-10, :, :] = image_2d #裁剪到16维度

    # 旋转
    random_rotate_type = random.randint(0, 359)
    people_image = rotate_3d(people_image,random_rotate_type)

    # 翻转
    random_flip_type = random.randint(0,3)
    people_image = flip(people_image,random_flip_type)

    # 亮度变化
    random_gray_type = random.randint(0,2)
    if random_gray_type == 0:
        people_image = people_image
    elif random_gray_type == 1:
        people_image = people_image * 0.8
    else:
        people_image = people_image * 1.2

    # for i in range(16):
    #     plt.figure("jieguo")
    #     plt.imshow(people_image[i,:,:])
    #     plt.pause(0.2)
        # plt.close("jieguo")
    return people_image
def read_nii_file_val(img_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = sitk.ReadImage(img_path)
    image_arr = sitk.GetArrayFromImage(nii_image)
    image_arr_back = norm_img(image_arr[0, 150:342, 30:222])  # 裁剪到192*192
    return image_arr_back
def people_image_train_val(people_dir): #截取输入数据到（32，224，224）
    files_singles = os.listdir(people_dir)
    people_image = np.zeros(shape=(16, 192, 192))
    for f in files_singles:
        image_2d = read_nii_file_val(os.path.join(people_dir, f)) # 读取图片
        if int(f) >= 10 and int(f) < 26:
            people_image[int(f)-10, :, :] = image_2d #裁剪到32维度
    return people_image
def people_label(): #加载表格（记录病例滑膜炎的等级）
    excel_label_dir = './IPFP MOAKS score.xlsx'
    book = pandas.read_excel(excel_label_dir, engine='openpyxl', sheet_name='Sheet2')
    sheets2 = book.values
    return sheets2

def zongimage_validate():
    zong_image = []
    zong_label = []
    files_peoples = os.listdir(NII_DIR_validate)
    for f in files_peoples:
        files_slice = os.listdir(os.path.join(NII_DIR_validate, f))
        files_slice.sort(key=lambda x: int(x[0:3]))
        ####删除非IPFP标签
        for i_dele in range(5):
            temp_dele = files_slice[i_dele - 5]
            if temp_dele[-3:] == 'nii':
                if temp_dele[-8:-4] != 'IPFP':
                    print(files_slice.pop(i_dele - 5))
        mri_label = read_label(os.path.join(NII_DIR_validate, f), files_slice[-1])  # 读取标签
        for si in files_slice[0:-1]:
            mri_label_singe = mri_label[mri_label.shape[0] - int(si)]  # 选择对应的标签
            image_2d = read_nii_file(os.path.join(NII_DIR_validate, f, si))  # 读取图片

            biaoji = 0
            '''
            if int(si) > 5 and int(si) < mri_label.shape[0] - 5:
                biaoji = 1
            '''
            if np.max(mri_label_singe) > 0:
                biaoji = 1

            if biaoji != 0:
                zong_image.append(image_2d)
                zong_label.append(mri_label_singe)
                '''
                plt.figure("jieguo")
                plt.subplot(211)
                plt.imshow(image_2d)
                plt.pause(0.01)

                plt.subplot(212)
                plt.imshow(mri_label_singe)
                plt.pause(1)
                plt.close("jieguo")
                '''
    return zong_image, zong_label

if __name__ == '__main__':
    people_dir = 'D:\BaiduNetdiskDownload/clc_data/9028418/9028418-00m-SAG_IW_TSE_RIGHT'
    # img = cv2.imread('./zidane.jpg')
    # img_rotate = rotate(img,39)
    # cv2.imshow('roatate',img_rotate)
    # cv2.waitKey(0)
    # #people_dir = 'D:\BaiduNetdiskDownload/clc_data/9002430/9002430-00m-SAG_IW_TSE_RIGHT'
    people_label()
    people_image_train(people_dir)
