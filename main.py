import cv2
import math
import numpy as np
from numpy.core.fromnumeric import shape
from tqdm import tqdm

def matching_100(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    tmp = cv2.imread( "../data/100/100-Template.jpg", cv2.IMREAD_GRAYSCALE)
    output_img = np.zeros((img.shape[0]-tmp.shape[0]+1, img.shape[1]-tmp.shape[1]+1))
    print(tmp.shape)
    # opencv
    # w, h = tmp.shape[::-1]
    # res = cv2.matchTemplate(img,tmp,cv2.TM_CCOEFF_NORMED)
    # loc = np.where( res >= 0.7)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
    # show(img)


    #cv2.circle(tmp, (85,126), 2, (0,0,255), 5, 16)
    #show(tmp, "tmp")
    resize_ratio = 16
    thres = 0.6
    resize_img = cv2.resize(img, (int(img.shape[1]/resize_ratio), int(img.shape[0]/resize_ratio)))
    resize_tmp = cv2.resize(tmp, (int(tmp.shape[1]/resize_ratio), int(tmp.shape[0]/resize_ratio)))
    resize_output = np.zeros((resize_img.shape[0]-resize_tmp.shape[0]+1, resize_img.shape[1]-resize_tmp.shape[1]+1))
    resize_matching = get_matching_result(resize_img, resize_tmp, resize_output)
    y_range, x_range = np.where(resize_matching>thres)
    end_x, start_x = max(x_range)*resize_ratio, min(x_range)*resize_ratio
    end_y, start_y = max(y_range)*resize_ratio, min(y_range)*resize_ratio
    # 若是範圍超過則取影像最大值
    if start_x<0:
        start_x = 0
    elif start_y<0:
        start_y = 0
    cv2.circle(img, (start_x, start_y), 3, (255,255,255), 5, 16)
    cv2.circle(img, (end_x, end_y), 3, (255,255,255), 5, 16)
    show(img, "IM")

def get_matching_result(img, tmp, output_img):
    tmp_u = np.mean(tmp)
    for i in tqdm(range(output_img.shape[0])):
        for j in range(output_img.shape[1]):
            #tmplate
            inner = 0
            norm_tmp = 0
            norm_img = 0
            # 計算符合template大小的原始影像平均值
            img_sum = [img[i+ti,j+tj] for ti in range(tmp.shape[0]) for tj in range(tmp.shape[1])]
            img_u = np.mean(img_sum)
            for ti in range(tmp.shape[0]):
                for tj in range(tmp.shape[1]):
                    t = tmp[ti, tj] - tmp_u
                    im = img[i+ti,j+tj] - img_u
                    inner += t*im 
                    norm_tmp += np.power(t, 2)
                    norm_img += np.power(im, 2)
            output_img[i, j]= (inner)/np.sqrt(norm_img*norm_tmp)
    
    return output_img


def show(image, Window_name="Image"):
    # 顯示圖片
    cv2.imshow(Window_name, image)

    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def angle(edge):
    edge_dx, edge_dy = edge[0]-edge[1], edge[2]-edge[3]
    edge_ang = int(math.atan2(edge_dx, edge_dy) * 180/math.pi)

    return 

if __name__=="__main__":
    matching_100("../data/100/100-1.jpg")