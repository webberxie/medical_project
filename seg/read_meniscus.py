import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from skimage import morphology
import imgaug.augmenters as iaa
import SimpleITK as sitk
from PIL import Image
import cv2
import random
NII_DIR_DATA='E:/medical_label/ROI(IOA)/image_data'
NII_DIR_LABEL='E:/medical_label/ROI(IOA)/ROI-WEN_new'
NII_DIR_LABEL_NEW = 'E:/medical_label/ROI(IOA)/ROI-WEN_label'
def norm_img(image): # 归一化像素值到（0，255）之间，且将溢出值取边界值
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

# 读入nii图像数据，返回归一化后的数组
def read_nii_file(img_path):
    '''
    根据nii文件路径读取nii图像
    '''
    nii_image = sitk.ReadImage(img_path)
    image_arr = sitk.GetArrayFromImage(nii_image)
    # 对图像进行归一化
    # image_arr_back = norm_img(image_arr)
    image_arr_back = norm_img(image_arr[0,112:336,110:334])
    return image_arr_back

# 读入标注图像数据，返回一个数组
def read_label(file_path, file):
    nii_image = sitk.ReadImage(os.path.join(file_path, file))
    image_arr = sitk.GetArrayFromImage(nii_image)
    # return image_arr
    return image_arr[:,112:336,110:334]

def check_has_label(col):
    for index in col:
        if index == 1 or index ==2:
            return True,index
    return False,0

def merge(new_list,del_list,flag_list):
    # 将删除的区域就近合并
    if len(new_list) == 4:
        center1 = (new_list[0] + new_list[1])/2
        center2 = (new_list[2] + new_list[3])/2
        for num in del_list:
            if abs(flag_list[num] - center1) < abs(flag_list[num]-center2):
                if flag_list[num] < flag_list[0]:
                    new_list[0] = flag_list[num]
                if flag_list[num] >flag_list[1]:
                    new_list[1] = flag_list[num]
            else:
                if flag_list[num] < new_list[2]:
                    new_list[2] = flag_list[num]
                if flag_list[num] >new_list[3]:
                    new_list[3] = flag_list[num]
    if len(new_list) == 2:
        for num in del_list:
            if flag_list[num] < new_list[0]:
                new_list[0] = flag_list[num]
            if flag_list[num] >new_list[1]:
                new_list[1] = flag_list[num]
    return new_list



def check_label2d_connect(label_2d):
    # 对于一个矢状面，统计是否连通,以及边界位置,标注类型
    img_h, img_w = label_2d.shape
    flag ,flag_list ,label_type = 0, [] ,0
    change_flag = False
    for col in range(img_w - 1):
        has_label1,labeltype1 = check_has_label(label_2d[:, col])
        has_label2,labeltype2 = check_has_label(label_2d[:, col + 1])
        # 判断标注类型
        if not change_flag:
            if labeltype1 != 0:
                label_type = labeltype1
                change_flag = True
        # 统计边界
        if ( has_label1 ^ has_label2) == True:
            flag += 1
            flag_list.append(col)
    # print(slice, '变化次数：', str(flag))

    del_list = []
    if len(flag_list) != 0:
        # 分析是否混入杂点，去除杂点
        for i in range(len(flag_list)//2):
            if abs( flag_list[i*2] - flag_list[i*2+1] ) < 4:
                del_list.append(2*i)
                del_list.append(2 * i+1)

        new_list = [flag_list[i] for i in range(len(flag_list)) if i not in del_list]
        if len(del_list)!=0:
            new_list = merge(new_list,del_list,flag_list)
        flag_list = new_list



    if len(flag_list) == 0:
        return 'nolabel',[],0
    elif len(flag_list) == 2:
        return 'connected',flag_list,label_type
    elif len(flag_list) == 4:
        return 'splited',flag_list,label_type
    else:
        raise ValueError('flag_list长度非4')

# 完成单层(矢状位)的自动分割
def change_label(label2d,is_connect,flag_list,label_type,single_flag):
    img_h, img_w = label2d.shape
    if label_type == 1:
        # 外侧
        if is_connect == 'connected':
            # 连通,为体部
            for col in range(flag_list[0],flag_list[1]+1):
                for row in range(img_h):
                    if label2d[row,col] == 1:
                        label2d[row,col] = 2
        if is_connect == 'splited':
            if single_flag == 'back_angle':
                # 只有一角，为后角
                for col in range(flag_list[0], flag_list[1] + 1):
                    for row in range(img_h):
                        if label2d[row, col] == 1:
                            label2d[row, col] = 1
            elif single_flag == 'head_angle':
                # 只有一角，为前角
                for col in range(flag_list[0], flag_list[1] + 1):
                    for row in range(img_h):
                        if label2d[row, col] == 1:
                            label2d[row, col] = 3
            else:
                # 不连通，分为后角，前角
                # 后角
                for col in range(flag_list[0],flag_list[1]+1):
                    for row in range(img_h):
                        if label2d[row,col] == 1:
                            label2d[row,col] = 1
                # 前角
                for col in range(flag_list[2],flag_list[3]+1):
                    for row in range(img_h):
                        if label2d[row,col] == 1:
                            label2d[row,col] = 3
    if label_type == 2:
        # 内侧
        if is_connect == 'connected':
            # 连通,为体部
            for col in range(flag_list[0], flag_list[1] + 1):
                for row in range(img_h):
                    if label2d[row, col] == 2:
                        label2d[row, col] = 5
        if is_connect == 'splited':
            if single_flag == 'back_angle':
                # 只有一角，为后角
                for col in range(flag_list[0], flag_list[1] + 1):
                    for row in range(img_h):
                        if label2d[row, col] == 2:
                            label2d[row, col] = 4
            elif single_flag == 'head_angle':
                # 只有一角，为前角
                for col in range(flag_list[0], flag_list[1] + 1):
                    for row in range(img_h):
                        if label2d[row, col] == 2:
                            label2d[row, col] = 6
            else:
                # 不连通，分为后角，前角
                # 后角
                for col in range(flag_list[0], flag_list[1] + 1):
                    for row in range(img_h):
                        if label2d[row, col] == 2:
                            label2d[row, col] = 4
                # 前角
                for col in range(flag_list[2], flag_list[3] + 1):
                    for row in range(img_h):
                        if label2d[row, col] == 2:
                            label2d[row, col] = 6

    return label2d


def auto_seg():
    # 自动分割
    allowed_state=['nolabel','connected','splited','nolabel','splited','connected','nolabel']

    files_peoples = os.listdir(NII_DIR_DATA)

    for files_people in files_peoples:
    # files_people = files_peoples[2]
        state_num = 0
        single_flag = 'none'
        last_flag_list = []
        files_people_num = files_people[0:7]
        if files_people_num != '9100862':
            continue

        files_slice = os.listdir(os.path.join(NII_DIR_DATA, files_people))
        mri_label = read_label(NII_DIR_LABEL, files_people_num+'.nii')    #读取标签(整体，多层)
        mri_label_new = np.zeros(shape=( 37, 444, 448),dtype=np.int8)
        for slice in files_slice:

            # image_2d = read_nii_file(os.path.join(NII_DIR_DATA, files_people, slice))
            label_2d = mri_label[mri_label.shape[0] - int(slice)]

            is_connect,flag_list,label_type = check_label2d_connect(label_2d)


            # print(is_connect,flag_list,label_type)
            if state_num != 6:
                # 检查是否存在状态错误
                if is_connect == allowed_state[state_num]:
                    pass
                elif is_connect == allowed_state[state_num+1]:
                    state_num += 1
                else:
                    # 当前为splited,误判为connected;纠正为splited;
                    if state_num == 2:
                        is_connect = allowed_state[state_num]
                        # 定位当前区域是前角还是后角
                        center1 = (last_flag_list[0] + last_flag_list[1])/ 2
                        center2 = (last_flag_list[2] + last_flag_list[3])/ 2
                        temp_center = (flag_list[0] + flag_list[1]) /2
                        if abs(temp_center - center1) < abs(temp_center - center2):
                            # 距离后角更近，为后角
                            single_flag = 'back_angle'
                        else:
                            # 距离前角更近，为前角
                            single_flag = 'head_angle'
                    # 当前为nolabel,突变为splited,误判为connected;纠正为splited
                    elif state_num == 3:
                        is_connect = allowed_state[state_num+1]
                        # 定位当前区域是前角还是后角
                        center1 = (last_flag_list[0] + last_flag_list[1]) / 2
                        center2 = (last_flag_list[2] + last_flag_list[3]) / 2
                        temp_center = (flag_list[0] + flag_list[1]) / 2
                        if abs(temp_center - center1) < abs(temp_center - center2):
                            # 距离后角更近，为后角
                            single_flag = 'back_angle'
                        else:
                            # 距离前角更近，为前角
                            single_flag = 'head_angle'


            if is_connect != 'nolabel':
                label_2d = change_label(label_2d, is_connect, flag_list, label_type,single_flag)
                single_flag = 'none'
                if len(flag_list) == 4:
                    last_flag_list = flag_list

            mri_label_new[mri_label.shape[0] - int(slice),:,:] = label_2d



        out = sitk.GetImageFromArray(mri_label_new)
        sitk.WriteImage(out,os.path.join(NII_DIR_LABEL_NEW,files_people_num+'.nii'))
        print(files_people_num,"finished")

        # 统计灰度
        # dict_hist={}
        # x,y= label_2d.shape
        # for i in range(x):
        #     for j in range(y):
        #        if label_2d[i,j] in dict_hist.keys():
        #            dict_hist[label_2d[i,j]] += 1
        #        else:
        #            dict_hist[label_2d[i, j]] = 1

    '''
        plt.figure("jieguo")
        plt.subplot(211)
        plt.imshow(image_2d)
        # plt.pause(2)
    
        plt.subplot(212)
        plt.imshow(label_2d)
    
        # imhist, bins = np.histogram(label_2d.flatten(), 256)
        #
        # plt.subplot(313)
        # plt.plot(imhist)
    
        # print(dict_hist)
        plt.pause(0.1)
        plt.close("jieguo")
    '''

def muti_class_label_transform(label_2d):
    muti_label = np.zeros(shape=(2,224,224))
    for i in range(224):
        for j in range(224):
            if label_2d[i,j] == 1:
                muti_label[0,i,j] = 1
            if label_2d[i,j] == 2:
                muti_label[1,i,j] = 1
    return muti_label

def single_class_label_transform(label_2d):
    for i in range(224):
        for j in range(224):
            if label_2d[i,j] == 2:
                label_2d[i,j] = 1
    return label_2d


def zongimage_train(enhance_move=False,enhance_elastic=False):
    zong_image = []
    zong_label = []
    files_peoples = os.listdir(NII_DIR_DATA)
    # 定义数据增强策略
    seq_Affine = iaa.Sequential([
        iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}),  #平移
        # iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0) # 控制点方式做形变
        # iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0))
    ])
    # 定义数据增强
    seq_PiecewiseAffine = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}),  # 平移
        iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0) # 控制点方式做形变
        # iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0))
    ])
    # 定义数据增强
    seq_ElasticTransformation = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}),  # 平移
        # iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0)  # 控制点方式做形变
        iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0))
    ])
    for files_people in files_peoples:
        files_people_num = files_people[0:7]
        if files_people_num not in [ '9100862','9095103']:
            files_slice = os.listdir(os.path.join(NII_DIR_DATA, files_people))
            mri_label = read_label(NII_DIR_LABEL, files_people_num + '.nii')  # 读取标签(整体，多层)
            for slice in files_slice:
                if int(slice)>=5 and int(slice)<=mri_label.shape[0]-5:
                    # 对于每一层
                    image_2d = read_nii_file(os.path.join(NII_DIR_DATA, files_people, slice))
                    label_2d = mri_label[mri_label.shape[0] - int(slice)]

                    # 加通道
                    image_2d = np.expand_dims(image_2d, axis=(0,3)).astype(np.float32) # 尤其注意这里数据格式 (batch_size, H, W, C)
                    label_2d = np.expand_dims(label_2d, axis=(0,3)).astype(np.int32)  # segmentation 需要时 int 型

                    # 增强
                    if enhance_move:
                        affine_image_2d, affine_label_2d = seq_Affine(images=image_2d, segmentation_maps=label_2d)
                    if enhance_elastic:
                        PiecewiseAffine_image_2d, PiecewiseAffine_label_2d = seq_Affine(images=image_2d, segmentation_maps=label_2d)
                        ElasticTransformation_image_2d, ElasticTransformation_label_2d = seq_Affine(images=image_2d, segmentation_maps=label_2d)

                    # 去通道
                    image_2d = np.squeeze(image_2d,axis=(0,3))
                    label_2d = np.squeeze(label_2d, axis=(0, 3))
                    if enhance_move:
                        affine_image_2d = np.squeeze(affine_image_2d, axis=(0, 3))
                        affine_label_2d = np.squeeze(affine_label_2d, axis=(0, 3))
                    if enhance_elastic:
                        PiecewiseAffine_image_2d = np.squeeze(PiecewiseAffine_image_2d, axis=(0, 3))
                        PiecewiseAffine_label_2d = np.squeeze(PiecewiseAffine_label_2d, axis=(0, 3))
                        ElasticTransformation_image_2d = np.squeeze(ElasticTransformation_image_2d, axis=(0, 3))
                        ElasticTransformation_label_2d = np.squeeze(ElasticTransformation_label_2d, axis=(0, 3))

                    # 标签转化为1分类
                    label_2d = single_class_label_transform(label_2d)
                    if enhance_move:
                        affine_label_2d = single_class_label_transform(affine_label_2d)
                    if enhance_elastic:
                        PiecewiseAffine_label_2d = single_class_label_transform(PiecewiseAffine_label_2d)
                        ElasticTransformation_label_2d = single_class_label_transform(ElasticTransformation_label_2d)

                    zong_image.append(image_2d)
                    zong_label.append(label_2d)
                    if enhance_move:
                        zong_image.append(affine_image_2d)
                        zong_label.append(affine_label_2d)
                    if enhance_elastic:
                        zong_image.append(PiecewiseAffine_image_2d)
                        zong_label.append(PiecewiseAffine_label_2d)
                        zong_image.append(ElasticTransformation_image_2d)
                        zong_label.append(ElasticTransformation_label_2d)

                    # plt.figure("jieguo")
                    # plt.subplot(221)
                    # plt.imshow(image_2d)
                    #
                    # plt.subplot(222)
                    # plt.imshow(label_2d)
                    #
                    # plt.subplot(223)
                    # plt.imshow(new_image_2d)
                    #
                    # plt.subplot(224)
                    # plt.imshow(new_label_2d)
                    # plt.pause(0.1)
                    # plt.close("jieguo")
    return zong_image, zong_label


def zongimage_validate(enhance_move=False,enhance_elastic=False):
    zong_image = []
    zong_label = []
    files_peoples = os.listdir(NII_DIR_DATA)
    # 定义数据增强策略
    seq_Affine = iaa.Sequential([
        iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}),  # 平移
        # iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0) # 控制点方式做形变
        # iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0))
    ])
    # 定义数据增强
    seq_PiecewiseAffine = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}),  # 平移
        iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0)  # 控制点方式做形变
        # iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0))
    ])
    # 定义数据增强
    seq_ElasticTransformation = iaa.Sequential([
        # iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}),  # 平移
        # iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0)  # 控制点方式做形变
        iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0))
    ])
    for files_people in files_peoples:
        files_people_num = files_people[0:7]
        if files_people_num in ['9100862', '9095103']:
            files_slice = os.listdir(os.path.join(NII_DIR_DATA, files_people))
            mri_label = read_label(NII_DIR_LABEL, files_people_num + '.nii')  # 读取标签(整体，多层)
            for slice in files_slice:
                # 对于每一层
                if int(slice) >= 5 and int(slice) <= mri_label.shape[0] - 5:
                    image_2d = read_nii_file(os.path.join(NII_DIR_DATA, files_people, slice))
                    label_2d = mri_label[mri_label.shape[0] - int(slice)]

                    # 加通道
                    image_2d = np.expand_dims(image_2d, axis=(0, 3)).astype(np.float32)  # 尤其注意这里数据格式 (batch_size, H, W, C)
                    label_2d = np.expand_dims(label_2d, axis=(0, 3)).astype(np.int32)  # segmentation 需要时 int 型

                    # 增强
                    if enhance_move:
                        affine_image_2d, affine_label_2d = seq_Affine(images=image_2d, segmentation_maps=label_2d)
                    if enhance_elastic:
                        PiecewiseAffine_image_2d, PiecewiseAffine_label_2d = seq_Affine(images=image_2d,
                                                                                        segmentation_maps=label_2d)
                        ElasticTransformation_image_2d, ElasticTransformation_label_2d = seq_Affine(images=image_2d,
                                                                                                    segmentation_maps=label_2d)

                    # 去通道
                    image_2d = np.squeeze(image_2d, axis=(0, 3))
                    label_2d = np.squeeze(label_2d, axis=(0, 3))
                    if enhance_move:
                        affine_image_2d = np.squeeze(affine_image_2d, axis=(0, 3))
                        affine_label_2d = np.squeeze(affine_label_2d, axis=(0, 3))
                    if enhance_elastic:
                        PiecewiseAffine_image_2d = np.squeeze(PiecewiseAffine_image_2d, axis=(0, 3))
                        PiecewiseAffine_label_2d = np.squeeze(PiecewiseAffine_label_2d, axis=(0, 3))
                        ElasticTransformation_image_2d = np.squeeze(ElasticTransformation_image_2d, axis=(0, 3))
                        ElasticTransformation_label_2d = np.squeeze(ElasticTransformation_label_2d, axis=(0, 3))

                    # 标签转化为1分类
                    label_2d = single_class_label_transform(label_2d)
                    if enhance_move:
                        affine_label_2d = single_class_label_transform(affine_label_2d)
                    if enhance_elastic:
                        PiecewiseAffine_label_2d = single_class_label_transform(PiecewiseAffine_label_2d)
                        ElasticTransformation_label_2d = single_class_label_transform(ElasticTransformation_label_2d)

                    zong_image.append(image_2d)
                    zong_label.append(label_2d)
                    if enhance_move:
                        zong_image.append(affine_image_2d)
                        zong_label.append(affine_label_2d)
                    if enhance_elastic:
                        zong_image.append(PiecewiseAffine_image_2d)
                        zong_label.append(PiecewiseAffine_label_2d)
                        zong_image.append(ElasticTransformation_image_2d)
                        zong_label.append(ElasticTransformation_label_2d)

                    # plt.figure("jieguo")
                    # plt.subplot(221)
                    # plt.imshow(image_2d)
                    #
                    # plt.subplot(222)
                    # plt.imshow(label_2d)
                    #
                    # plt.subplot(223)
                    # plt.imshow(new_image_2d)
                    #
                    # plt.subplot(224)
                    # plt.imshow(new_label_2d)
                    # plt.pause(0.1)
                    # plt.close("jieguo")
    return zong_image, zong_label


def preprocess():
    # 读取训练数据集
    train_image, train_label = zongimage_train(enhance_move=True)
    # 进行随机打乱
    random_seed_train = random.sample(range(0, len(train_image)), len(train_image))
    random_train_image = train_image.copy()
    random_train_label = train_label.copy()

    for i in range(len(train_image)):
        random_train_image[i] = train_image[random_seed_train[i]].copy()
        random_train_label[i] = train_label[random_seed_train[i]].copy()
    np.savez('datasets_meniscus_train', suiji_train_image=random_train_image, suiji_train_label=random_train_label)  # 保存乱序后的数据集

    # 读取验证集数据
    validate_image, validate_label = zongimage_validate(enhance_move=True)
    # 进行随机打乱
    random_seed_validate = random.sample(range(0, len(validate_image)), len(validate_image))
    random_validate_image = validate_image.copy()
    random_validate_label = validate_label.copy()

    for i in range(len(validate_image)):
        random_validate_image[i] = validate_image[random_seed_validate[i]].copy()
        random_validate_label[i] = validate_label[random_seed_validate[i]].copy()
    np.savez('datasets_meniscus_validate', suiji_validate_image=random_validate_image,
             suiji_validate_label=random_validate_label)  # 保存乱序后的数据集

def label_adjust():
    files_peoples = os.listdir(NII_DIR_DATA)

    for files_people in files_peoples:
        files_people_num = files_people[0:7]
        if files_people_num == '9004669':
            continue

        files_slice = os.listdir(os.path.join(NII_DIR_DATA, files_people))
        mri_label = read_label(NII_DIR_LABEL, files_people_num + '.nii')  # 读取标签(整体，多层)
        mri_label_new = np.zeros(shape=(38, 444, 448), dtype=np.int8)
        for slice in files_slice:
            label_2d = mri_label[mri_label.shape[0] - int(slice)]
            for i in range(444):
                for j in range(448):
                    if label_2d[i, j] == 4:
                        label_2d[i, j] = 1
                    elif label_2d[i, j] == 1:
                        label_2d[i, j] = 2
            mri_label_new[mri_label.shape[0] - int(slice), :, :] = label_2d
        out = sitk.GetImageFromArray(mri_label_new)
        sitk.WriteImage(out, os.path.join(NII_DIR_LABEL_NEW, files_people_num + '.nii'))
        print(files_people_num, "finished")



if __name__ == '__main__':
    preprocess()
    # auto_seg()
    # label_adjust()
