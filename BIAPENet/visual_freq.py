import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("test1.png",0)
# 数据类型转换 转换为浮点型
print('0\n',img)
img1 = img.astype(np.float32)


# 进行离散余弦变换
img_dct = cv2.dct(img1)
print('1\n',img_dct)
# 进行log处理
img_dct_log = np.log(abs(img_dct))
print('2\n',img_dct_log)
# 进行离散余弦反变换
img_idct = cv2.idct(img_dct)
print('3\n',img_idct)
# res = img_idct.astype(np.uint8) # 浮点型转整型 小数部分截断
# print('3-1\n',res)

plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.subplot(132)
plt.imshow(img_dct_log, cmap="gray")
plt.title('DCT'),plt.xticks([]),plt.yticks([])
plt.subplot(133)
plt.imshow(img_idct, cmap="gray")
plt.title('IDCT'),plt.xticks([]),plt.yticks([])
plt.show()

# coding=utf-8

# # 底板图案
# bottom_pic = r'E:\cbd\ICCV\improve_yolo\1.png'
# # 上层图案
# top_pic = r'E:\cbd\ICCV\improve_yolo\2.png'
#
# import cv2
# bottom = cv2.imread(bottom_pic)
# bottom = cv2.resize(bottom, (128, 128 ))
# top = cv2.imread(top_pic)
# top = cv2.resize(top, (128 ,128))
# # 权重越大，透明度越低
# overlapping = cv2.addWeighted(bottom, 0.2, top, 0.2, 0)
# # 保存叠加后的图片
# cv2.imwrite('result.jpg', overlapping)