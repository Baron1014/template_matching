import cv2
import math
import numpy as np
from numpy.core.fromnumeric import shape
from tqdm import tqdm
from datetime import datetime

def matching_100(img_path, thres=0.5, rot=0, scale=1, method=None):
    file_name = img_path.split('/')[-1].replace(".jpg", "")
    start = datetime.now()

    # read image & template
    plot_img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    tmp = cv2.imread( "data/100/100-Template.jpg", cv2.IMREAD_GRAYSCALE)

    # Template Rotation
    tmp = rot_image(tmp, rot, scale)

    if method=="opencv":
        res = cv2.matchTemplate(img,tmp,cv2.TM_CCOEFF_NORMED)
        # small search 
        local_max_points = find_opencv_local_max(res, thres)

        print(f"{img_path.split('/')[-1]} doing Texture Matching with OpenCV cost: {datetime.now()-start}")
    else:
        # Pyramid Down
        resize_ratio = 16
        resize_img = cv2.resize(img, (int(img.shape[1]/resize_ratio), int(img.shape[0]/resize_ratio)))
        resize_tmp = cv2.resize(tmp, (int(tmp.shape[1]/resize_ratio), int(tmp.shape[0]/resize_ratio)))
        
        # Texture Matching
        resize_output = np.zeros((resize_img.shape[0]-resize_tmp.shape[0]+1, resize_img.shape[1]-resize_tmp.shape[1]+1))
        resize_matching = get_matching_result(resize_img, resize_tmp, resize_output)
        
        # set threshold and get origin points
        points = [i for i in zip(np.where(resize_matching>thres))]
        mapping_origin_points = [i[0]*resize_ratio for i in points]

        # small search 
        local_max_points = get_small_matching_result(img, tmp, mapping_origin_points, resize_ratio)
        print(f"{img_path.split('/')[-1]} doing Texture Matching cost: {datetime.now()-start}")
    

    # plot bounding box
    for i in range(len(local_max_points)):
        y, x = local_max_points[i][0]
        score = local_max_points[i][1]
        center_x, center_y = x+(int(tmp.shape[1]/2)), y+(int(tmp.shape[0]/2))
        cv2.rectangle(plot_img, (x,y), (x+tmp.shape[1], y+tmp.shape[0]), (255,0,0) if method=="opencv" else (0,0,255) , 4)
        cv2.circle(plot_img, (center_x, center_y), 1, (0,255,255), 5, 16)
        cv2.putText(plot_img, f"X:{center_x}", (center_x+30, center_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Y:{center_y}", (center_x+30, center_y+25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Scale:{scale}", (center_x+30, center_y+50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Angle:{rot}", (center_x+30, center_y+75), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Score:{score:.3f}", (center_x+30, center_y+100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    #show(plot_img, "Image")
    save_image(plot_img, file_name + f"_matching{f'_{method}' if method is not None else str()}", "data/100/")

def matching_die(img_path, thres=0.8, rot=0, scale=1, method=None):
    file_name = img_path.split('/')[-1].replace(".tif", "")
    start = datetime.now()

    # read image & template
    plot_img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    tmp = cv2.imread( "data/Die/Die-Template.tif", cv2.IMREAD_GRAYSCALE)
    
    # Template Rotation
    tmp = rot_image(tmp, rot, scale)
    if method=="opencv":
        res = cv2.matchTemplate(img,tmp,cv2.TM_CCOEFF_NORMED)
        # small search 
        local_max_points = find_opencv_local_max(res, thres)

        print(f"{img_path.split('/')[-1]} doing Texture Matching with OpenCV cost: {datetime.now()-start}")
    else:
        # Pyramid Down
        resize_ratio = 8
        resize_img = cv2.resize(img, (int(img.shape[1]/resize_ratio), int(img.shape[0]/resize_ratio)))
        resize_tmp = cv2.resize(tmp, (int(tmp.shape[1]/resize_ratio), int(tmp.shape[0]/resize_ratio)))

        # Texture Matching
        resize_output = np.zeros((resize_img.shape[0]-resize_tmp.shape[0]+1, resize_img.shape[1]-resize_tmp.shape[1]+1))
        resize_matching = get_matching_result(resize_img, resize_tmp, resize_output)
        
        # set threshold and get origin points
        points = [i for i in zip(np.where(resize_matching>thres))]
        mapping_origin_points = [i[0]*resize_ratio for i in points]

        # small search 
        local_max_points = get_small_matching_result(img, tmp, mapping_origin_points)
        print(f"{img_path.split('/')[-1]} doing Texture Matching cost: {datetime.now()-start}")
        

    # plot bounding box
    for i in range(len(local_max_points)):
        y, x = local_max_points[i][0]  
        score = local_max_points[i][1]
        center_x, center_y = x+(int(tmp.shape[1]/2)), y+(int(tmp.shape[0]/2))
        cv2.rectangle(plot_img, (x,y), (x+tmp.shape[1], y+tmp.shape[0]), (255,0,0) if method=="opencv" else (0,0,255), 1)
        cv2.circle(plot_img, (center_x, center_y), 1, (0,255,255), 5, 16)
        cv2.putText(plot_img, f"X:{center_x}", (center_x+10, center_y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Y:{center_y}", (center_x+10, center_y+15), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Scale:{scale}", (center_x+10, center_y+30), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Angle:{rot}", (center_x+10, center_y+45), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(plot_img, f"Score:{score:.3f}", (center_x+10, center_y+60), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    #show(plot_img, "Image")
    save_image(plot_img, file_name + f"_matching{f'_{method}' if method is not None else str()}", "data/Die/")

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


# 計算小範圍 matching
def get_small_matching_result(img, tmp, points, resize_ratio=8):
    tmp_u = np.mean(tmp)
    origin_points = list()
    for p in range(len(points[0])):
        y_min, y_max = points[0][p] - resize_ratio, points[0][p] + resize_ratio
        x_min, x_max = points[1][p] - resize_ratio, points[1][p] + resize_ratio
        local_max = 0
        for i in tqdm(range(y_min, y_max), desc=f"Small searching for ({points[0][p]},{points[1][p]})"):
            for j in range(x_min, x_max):
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
                local_value = (inner)/np.sqrt(norm_img*norm_tmp)
                if local_value > local_max:
                    local_max = local_value
                    best_point = (i, j)
        origin_points.append([best_point, local_max])
    return origin_points

# find opencv local maximum
def find_opencv_local_max(res, thres, ignore_number=30):
    points = [i[0] for i in zip(np.where(res>thres))]
    scores = res[res>thres]
    
    # init cluster
    c_x = list()
    ignore = 0
    for pc in range(res.shape[1]):
        if ignore==0:
            if pc in points[1]:
                c_x.append(pc)
                ignore = ignore_number
        else:
            ignore-=1
    # 按x先分群
    cluster = {i:[(0,0), 0] for i in c_x}
    for p in range(len(points[0])):
        y, x = points[0][p], points[1][p]
        s = scores[p]
        # init cluster
        for i in cluster.keys():
            if abs(x-i)<30 and s>cluster[i][1]:
                cluster[i] = [(y,x), s] 
    
    return [cluster[i] for i in cluster.keys()]
    

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

def rot_image(img, rot=0, scale=1):
    s = 1 + (1-scale)
    rot_img = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            adjust_i = i - (img.shape[0]//2)
            adjust_j = j - (img.shape[1]//2)
            x = int(s*(adjust_j*np.cos(np.radians(rot)) - adjust_i*np.sin(np.radians(rot))))
            y = int(s*(adjust_j*np.sin(np.radians(rot)) + adjust_i*np.cos(np.radians(rot))))
            re_x = x + (img.shape[1]//2)
            re_y = y + (img.shape[0]//2)
            if 0 <= re_x < img.shape[1]:
                if 0 <= re_y < img.shape[0]:
                    rot_img[i, j] = img[re_y, re_x]
                else:
                    rot_img[i, j] = 0
            else:
                rot_img[i, j] = 0

    return rot_img

def save_image(fig, figname, report_path):
    cv2.imwrite(f'{report_path}/{figname}.jpg', fig)

if __name__=="__main__":
    matching_100("data/100/100-1.jpg", thres=0.7, method="opencv")
    matching_100("data/100/100-2.jpg", thres=0.7, method="opencv")
    matching_100("data/100/100-3.jpg", thres=0.7, method="opencv")
    matching_100("data/100/100-4.jpg", thres=0.6, method="opencv")

    matching_100("data/100/100-1.jpg")
    matching_100("data/100/100-2.jpg", thres=0.4, rot=1)
    matching_100("data/100/100-3.jpg")
    matching_100("data/100/100-4.jpg")

    matching_die("data/Die/Die1.tif", thres=0.7, method="opencv")
    matching_die("data/Die/Die2.tif", thres=0.7, method="opencv")

    matching_die("data/Die/Die1.tif", thres=0.79, rot=2)
    matching_die("data/Die/Die2.tif", thres=0.747, rot=-2)