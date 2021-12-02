import cv2
import math
import numpy as np
from numpy.core.fromnumeric import shape
from tqdm import tqdm

def matching_100(img_path, thres=0.5, rot=0, scale=1):
    plot_img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    tmp = cv2.imread( "../data/100/100-Template.jpg", cv2.IMREAD_GRAYSCALE)
    output_img = np.zeros((img.shape[0]-tmp.shape[0]+1, img.shape[1]-tmp.shape[1]+1))
    print(plot_img.shape)
    # opencv
    # w, h = tmp.shape[::-1]
    # res = cv2.matchTemplate(img,tmp,cv2.TM_CCOEFF_NORMED)
    # loc = np.where( res >= 0.7)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
    #show(img)

    #opencv rotation
    center = (tmp.shape[1]//2,tmp.shape[0]//2)
    rot_mat = cv2.getRotationMatrix2D(center, rot, scale)
    tmp = cv2.warpAffine(tmp, rot_mat, (tmp.shape[1], tmp.shape[0]))
    #show(tmp)
    #cv2.circle(tmp, (85,126), 2, (0,0,255), 5, 16)
    # rot_tmp = np.zeros((tmp.shape[0], tmp.shape[1]))
    # for i in range(tmp.shape[0]):
    #     for j in range(tmp.shape[1]):
    #         x = int(j*np.cos(np.radians(5)) - i*np.sin(np.radians(5)))
    #         y = int(j*np.sin(np.radians(5)) + i*np.cos(np.radians(5)))
    #         if x>=rot_tmp.shape[1] or x<0:
    #             continue
    #         if y>=rot_tmp.shape[0] or y<0:
    #             continue
    #         rot_tmp[y, x] = tmp[i, j]
    #show(tmp)
    #show(rot_tmp, "tmp")
    resize_ratio = 16
    resize_img = cv2.resize(img, (int(img.shape[1]/resize_ratio), int(img.shape[0]/resize_ratio)))
    resize_tmp = cv2.resize(tmp, (int(tmp.shape[1]/resize_ratio), int(tmp.shape[0]/resize_ratio)))
    resize_output = np.zeros((resize_img.shape[0]-resize_tmp.shape[0]+1, resize_img.shape[1]-resize_tmp.shape[1]+1))
    resize_matching = get_matching_result(resize_img, resize_tmp, resize_output)
    points = [i for i in zip(np.where(resize_matching>thres))]
    sim_scores = resize_matching[resize_matching>thres]
    mapping_origin_points = [i[0]*resize_ratio for i in points]

    # plot bounding box
    for i in range(len(mapping_origin_points)):
        y = mapping_origin_points[0][i]
        x = mapping_origin_points[1][i]
        score = sim_scores[i]
        center_x, center_y = x+(int(tmp.shape[1]/2)), y+(int(tmp.shape[0]/2))
        cv2.rectangle(plot_img, (x,y), (x+tmp.shape[1], y+tmp.shape[0]), (0,0,255), 4)
        cv2.circle(plot_img, (center_x, center_y), 1, (0,255,255), 5, 16)
        cv2.putText(plot_img, f"X:{center_x}", (center_x+30, center_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Y:{center_y}", (center_x+30, center_y+25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Scale:{scale}", (center_x+30, center_y+50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Angle:{rot}", (center_x+30, center_y+75), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Score:{score:.3f}", (center_x+30, center_y+100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    show(plot_img, "IM")

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
    show(output_img)
    return output_img


def show(image, Window_name="Image"):
    # 顯示圖片
    cv2.namedWindow(Window_name,0)
    cv2.resizeWindow(Window_name, 1000, 800)
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
    matching_100("../data/100/100-2.jpg", thres=0.4, rot=1, scale=1.1)
    matching_100("../data/100/100-3.jpg")
    matching_100("../data/100/100-4.jpg")