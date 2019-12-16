# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:33:08 2019

@author: dogfi
"""
from cv2 import cv2
import numpy as np
import os
import time
import numba as nb


# 讀取圖像
def load_image(path, size_ratio=1):
    names = os.listdir(path)
    print('# Loading Images : ', names)
    data = []
    for name in names:
        image = cv2.imread(path + '/' + name)
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(size_ratio * w), int(size_ratio * h)))
        data.append(image)
        
    return data

# 圖像灰階處理
def cvt2gray(images):
    data = []
    for i in range(len(images)):
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        data.append(gray)
    
    return data

# 解析特徵點以及得出單應矩陣
def find_pts(kp_1, kp_2, good, MIN_MATCH_COUNT=4):
    # 確認是否有4個匹配點，單應矩陣需要至少4個點做計算
    if len(good) >= MIN_MATCH_COUNT:
        # 經由 篩選匹配結果解析 kp的 座標
        src_pts = np.float32([kp_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 解析好的兩組 kp 經由 RANSAC 隨機運算，得到最好的單應矩陣
        m = cv2.findHomography(src_pts,
                               dst_pts,
                               cv2.RANSAC,
                               5.0
                               )[0]

    else:
        print("Not enough matches are found - %d / %d" % (len(good), MIN_MATCH_COUNT))

    return m

# 篩選出最相近的特徵點
def find_good(matches, ratio=0.6):
    good = []
    for m in matches:
        if m[0].distance < ratio * m[1].distance:
            good.append(m[0])

    return good


# 建立特徵圖及匹配    
def surf_build(images, KK=1000, TH=0.4):

    # 建立特徵
    surf = cv2.xfeatures2d.SURF_create(KK, upright=True)
    # kp 及 des 為特徵點擷取後資訊
    kp1, des1 = surf.detectAndCompute(images[1], None)
    kp2, des2 = surf.detectAndCompute(images[0], None)
    
    # 建立匹配器相關功能
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 由特徵擷取後的des進行匹配
    matches = flann.knnMatch(des1, des2, k=2)
         
    good = find_good(matches, TH)
    print("# Matching pts : ", len(good))
    
    # 將篩選後結果與 kp (keypoint)計算出單應矩陣               
    data = find_pts(kp1, kp2, good) 
    
    return data
    

# 水平拼接
def merge_1(img1, img2, data, dst1):
    h, w = img1.shape[:2]
    # 將拼接圖經單應矩陣變換      
    merge = cv2.warpPerspective(img2,
                                data,
                                (dst1[3], img1.shape[0]),
                                flags=cv2.INTER_NEAREST
                                )
    # 複製變換後的拼接圖
    merge1 = merge.copy()
    tt = time.time()

    # 重疊處修正
    # 顯示 參考圖 及 拼接圖 的大小資訊
    print('# Reference image shape : ', img1.shape)  # 參考圖
    print('# Stitching image shape : ', merge.shape)  # 拼接圖
    merge = fix1_2(merge, merge1, img1, h, w, dst1)
    tto = time.time()
    print('# Fix Time : %.3f sec' % (tto - tt))

    return merge

# 三角權重羽化處理
@nb.njit(nogil=True)
def fix1_2(merge, merge1, img1, h, w, dst1):
    # 將參考圖填補回拼接圖
    for ch in range(3):
        for ro in range(img1.shape[0]):
            for co in range(img1.shape[1]):
                merge1[ro, co, ch] = img1[ro, co, ch]

    # 處理水平方向重疊處
    for ch in range(3):
        for c in range(dst1[4], w):
            for r in range(h):
                q = (np.pi * (w - c)) / (2 * (w - dst1[4]))

                # w_1 及 w_2 為權重
                w_1 = np.square(np.cos(q))
                w_2 = np.square(np.sin(q))

                # 三角權重改善演算法實現
                merge1[r, c, ch] = w_2 * img1[r, c, ch] + w_1 * merge[r, c, ch]

    return merge1

# 由單應矩陣解出座標點
def transform_pts(image, data):
    h, w = image.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, data)

    return dst, h, w


# 解析拼接圖點座標
def sol_dst(dst):
    start_row = int(dst[:, :, 1].min())
    start_col = int(dst[:, :, 0].min())
    end_row = int(dst[:, :, 1].max())
    end_col = int(dst[:, :, 0].max())
    start_sec_col = int(dst[:2, :, 0].max())

    return start_row, start_col, end_row, end_col, start_sec_col


if __name__ == '__main__':
    
    tic = time.time()
    
    images = load_image('./images/01/')    
    
    img_gray = cvt2gray(images)
    
    data = surf_build(img_gray)
    
    dst = transform_pts(images[0], data)[0]
    dst1 = sol_dst(dst)

    # 將單應矩陣 及 四座標點解析後，將拼接圖做變形並拼接參考圖
    merge1 = merge_1(img1=images[0],
                     img2=images[1],
                     data=data,
                     dst1=dst1
                     )
    
    toc = time.time()
    print('# Time : %.3f' % (toc-tic))
    cv2.imwrite('./outputs.png', merge1)
    



    