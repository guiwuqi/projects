import cv2
import numpy as np
from PIL import Image
import time


# 获取各个像素的概率
def getPixelProbability(gray):
    p_arr = np.zeros([256, ])    # 256个0.0
    gray = gray.ravel()          # 将图像转化为一维数组
    for p in gray:
        p_arr[p] += 1            # 像素出现一次计数加1
    return p_arr / len(gray)     # 返回概率


# 分离前景和背景
def separateForegroundAndBackground(arr, t):
    f_arr, b_arr = arr[:t], arr[t:]
    return f_arr, b_arr


# 获取某阈值下的信息熵
def informationEntropy(arr):
    Pn = np.sum(arr)
    if Pn == 0:
        return 0
    iE = 0
    for p_i in arr:
        if p_i == 0:
            continue
        iE += (p_i / Pn) * np.log(p_i / Pn)
    return -iE


# 获取最优阈值
def getMaxInformationEntropy(gray):
    p_arr = getPixelProbability(gray)
    IES = np.zeros([256, ])
    for t in range(1, len(IES)):
        f_arr, b_arr = separateForegroundAndBackground(p_arr, t)
        IES[t] = informationEntropy(f_arr) + informationEntropy(b_arr)
    return np.argmax(IES)


time_beginning = time.time()
image = cv2.imread('c.jpg')  # 读取图片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
T = getMaxInformationEntropy(gray)  # 获取阈值
ret, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
cv2.imshow("image", image)  # 可视化 原图
cv2.imwrite("gray.JPG", gray)
cv2.imshow("gray image", gray)  # 可视化 灰度图
cv2.imwrite("threshold.JPG", thresh)
cv2.imshow("threshold image", thresh)  # 可视化 阈值分割图
print(time.time()-time_beginning)
cv2.waitKey()


def hash_img(img):  # 计算图片的特征序列
    a = []  # 存储图片的像素
    hash_img = ''  # 特征序列
    width, height = 443, 591  # 图片缩放大小
    img = img.resize((width, height))  # 图片缩放为width×height
    for y in range(img.height):
        b = []
        for x in range(img.width):
            pos = x, y
            color_array = img.getpixel(pos)  # 获得像素
            # color = sum(color_array) / 3  # 灰度化
            b.append(int(color_array))
        a.append(b)
    for y in range(img.height):
        avg = sum(a[y]) / len(a[y])  # 计算每一行的像素平均值
        for x in range(img.width):
            if a[y][x] >= avg:  # 生成特征序列,如果此点像素大于平均值则为1,反之为0
                hash_img += '1'
            else:
                hash_img += '0'
    return hash_img


def similar(img1, img2):  # 求相似度
    hash1 = hash_img(img1)  # 计算img1的特征序列
    hash2 = hash_img(img2)  # 计算img2的特征序列
    differnce = 0
    for i in range(len(hash1)):
        differnce += abs(int(hash1[i]) - int(hash2[i]))
    similar = 1 - (differnce / len(hash1))
    return similar


img1 = Image.open('gray.JPG')
img2 = Image.open('threshold.JPG')
print('%.1f%%' % (similar(img1, img2) * 100))


