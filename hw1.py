import cv2
import numpy as np
# from google.colab.patches import cv2_imshow

img_org = cv2.imread("nchu_image.jpg",1)

# points1 = np.float32([[307,489], [512,479], [304,512], [516,512]])
# points2 = np.float32([[0,0], [200,0], [0,30], [200,30]])




# 原圖要抓的點
# p1 p2
# p3 p4
px1=307
py1=489
px2=512
py2=479
px3=304
py3=512
px4=510
py4=508

# 抓出長度及高度
top_left_x = min([px1,px2,px3,px4])
top_left_y = min([py1,py2,py3,py4])
bot_right_x = max([px1,px2,px3,px4])
bot_right_y = max([py1,py2,py3,py4])
len_x = bot_right_x-top_left_x
len_y = bot_right_y-top_left_y

# 轉換後的座標
x1=0
y1=0
x2=len_x
y2=0
x3=0
y3=len_y
x4=len_x
y4=len_y

# 創造出一個抓出長度集高度的空白矩陣
# 注意這邊是用Y,X
img = np.zeros((len_y, len_x, 3), dtype=img_org.dtype)  

data1 = np.array([[x1,y1,x1*y1,1,0,0,0,0],
           [x2,y2,x2*y2,1,0,0,0,0],
           [x3,y3,x3*y3,1,0,0,0,0],
           [x4,y4,x4*y4,1,0,0,0,0],
           [0,0,0,0,x1,y1,x1*y1,1],
           [0,0,0,0,x2,y2,x2*y2,1],
           [0,0,0,0,x3,y3,x3*y3,1],
           [0,0,0,0,x4,y4,x4*y4,1]])
data2 = np.array([[px1],
           [px2],
           [px3],
           [px4],
           [py1],
           [py2],
           [py3],
           [py4]])
# 求出八參數各個參數
result = np.linalg.solve(data1,data2)



# 再去跑每一個點相對應的數值
for i in range(len_x):
  for j in range(len_y):
    double_y = ((result[0] * i) + (result[1] * j) + (result[2] * i * j) + result[3])
    double_x = ((result[4] * i) + (result[5] * j) + (result[6] * i * j) + result[7])
    y = int(double_y)
    x = int(double_x)
    v = (double_y - y)
    u = (double_x - x)
    for c in range(3):
        # f(x,y) = (1-u)(1-v)g(x,y) + u(1-v)g(x,y+1) + v(1-u)g(x+1,y) + uvg(x+1,y+1)
        # 注意這邊是用Y,X,C
        img[j, i, c] = (1 - u) * (1 - v) * img_org[x, y][c] + \
                              u * (1 - v) * img_org[x, y + 1][c] + \
                              v * (1 - u) * img_org[x + 1, y][c] + \
                              u * v * img_org[x + 1, y + 1][c]


cv2.line(img_org, (307, 489), (512, 479), (0, 0, 255), 3)   #畫上面紅線
cv2.line(img_org, (307, 489), (304, 512), (0, 0, 255), 3)   #畫左邊紅線
cv2.line(img_org, (304, 512), (510, 508), (0, 0, 255), 3)   #畫下面紅線
cv2.line(img_org, (512, 479), (510, 508), (0, 0, 255), 3)   #畫下面紅線


cv2.imshow("原圖",img_org)
cv2.imshow("透視變形後的圖",img)

cv2.waitKey(0)