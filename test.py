import numpy as np
import math
import copy

A = [10,10]
B = [6,12]
B_ = [12,6]
C = [10,5]
C_ = [5,10]
D = [5,8]
D_ = [8,5]
name = ["A","B","C","D"]
flip = True

def trans(name):
    rect = []
    for i in name:
        if i == "A" :
            rect.append(A)
        if i == "A_": 
            rect.append(A_)
        if i == "B" :
            rect.append(B)
        if i == "B_": 
            rect.append(B_)
        if i == "C" :
            rect.append(C)
        if i == "C_" :
            rect.append(C_)
        if i == "D" :
            rect.append(D)
        if i == "D_" :
            rect.append(D_)
    return rect
    
def unpack_rect(rect,name):
    if name == "A" :
        return rect[0][0],rect[0][1]
    if name == "B" :
        return rect[1][0],rect[1][1]
    if name == "C" :
        return rect[2][0],rect[2][1]
    if name == "D" :
        return rect[3][0],rect[3][1] 
    if name == "B_" :
        return rect[1][0],rect[1][1]
    if name == "C_" :
        return rect[2][0],rect[2][1]
    if name == "D_" :
        return rect[3][0],rect[3][1] 

# make list sub sequence
def listExcludedIndices(data, indices=[]):
  return [x for i, x in enumerate(data) if i not in indices]

# make list
def make_parm(data,flip=False): 
    result = []
    for i in range(len(data)):
      for j in range(len(data) - 1):
        for k in range(len(data) - 2):
            for l in range(len(data) - 3):
                jData = listExcludedIndices(data, [i])
                kData = listExcludedIndices(jData, [j])
                lData = listExcludedIndices(kData, [k])
                result.append([data[i], jData[j], kData[k],lData[l]])

    return result

#　ganma+ と　ganma-の交差点を計算
def calc_flag(Gp,Gn):
    Gp_line = [Gp] * 4
    Gn_line = [Gn] * 4
    
    Gp_line = list(map(list,zip(*Gp_line)))
    flag = []
    for i in range(4):
        tmp = []
        for j in range(4):
            if Gp_line[i][j] == Gn_line[i][j]:
                tmp.append(1)
            else:
                tmp.append(0)
        flag.append(tmp)
    return np.array(flag)


#　Murata アルゴリズムの実行
def calc_best_placement(rect,flag,name,debug):
    flag_x = copy.copy(flag)
    weight_x = np.zeros((4,4))

    flag_y = copy.copy(flag)
    weight_y = np.zeros((4,4))

    debug_flag  = copy.copy(flag)
    print("--------------------------------------")

    #矩形の順番の計算
    for i in range(4):
        for j in range(4):
            flag_tmp = copy.copy(flag_x)
            if flag_x[j][i] >= 1:
                flag_tmp[j+1:,i:] = -1
                next_v = np.max(flag_x[:j+1,:i+1]) + 1
                

                flag_x = np.where((flag_tmp == -1) & (flag_x >= 1) & (flag_x <= next_v), next_v, flag_x) #+1ではなくxの幅#

    for j in  reversed(range(4)):
        for i in range(4):
            flag_tmp_y = copy.copy(flag_y)
            if flag_y[j][i] >= 1:
                flag_tmp_y[0:j,i:4] = -1
                next_v = np.max(flag_y[j:,:i+1]) + 1
                
                flag_y = np.where((flag_tmp_y == -1) & (flag_y >= 1) & (flag_y <= next_v), next_v, flag_y)

#長さ（重み）の計算
    for i in range(1,np.max(flag_x)):
        x,y = np.where(flag_x==i) # 一個前
        x_2,y_2 = np.where(flag_x==i + 1)# 見たいところ
        
        tmp = []
        tmp_map = []
        for j in x:
            rect_value_x,rect_value_y = unpack_rect(rect,name[j])
            tmp_map.append(np.where((flag_x == i + 1) , np.max(weight_x) + rect_value_x, weight_x)) #+1ではなくxの幅#
            tmp.append(np.max(tmp_map))
        
        index = np.argmax(tmp)
        weight_x = tmp_map[index]

    for i in range(1,np.max(flag_y)):
        x,y = np.where(flag_y==i) # 一個前
        x_2,y_2 = np.where(flag_y==i + 1)# 見たいところ
        tmp = []
        tmp_map = []
        for j in y:
            rect_value_x,rect_value_y = unpack_rect(rect,name[j])
            tmp_map.append(np.where((flag_y == i + 1) , np.max(weight_y) + rect_value_y, weight_y))
            tmp.append(np.max(tmp_map))
        index = np.argmax(tmp)
        weight_y = tmp_map[index]

#最終的なｘ，ｙの長さを計算
    x_max = []            
    x_last, _ = np.where(flag_x==np.max(flag_x))

    for j in x_last:
        rect_value_x,rect_value_y = unpack_rect(rect,name[j])
        x_max.append((np.max(weight_x)+rect_value_x))

    x_max_v = np.max(x_max)

    y_max = []            
    _, y_last = np.where(flag_y==np.max(flag_y))

    for j in y_last:
        rect_value_x,rect_value_y = unpack_rect(rect,name[j])
        y_max.append((np.max(weight_y)+rect_value_y))
    y_max_v = np.max(y_max)

    #面積の計算
    s = x_max_v*y_max_v

    return s,[x_max_v,y_max_v]

def main(name_list,rect):
    area_list = []
    place_list = []
    comb_list = [] 
    c = 0 
    for i in name_list:
        for j in name_list:
            print("=================time:",c,"========================")
            c += 1
            print("G+",i,"G-",j)
            flag = calc_flag(i,j)
            s,p = calc_best_placement(rect,flag,i,c)
            area_list.append(s)
            place_list.append(p)
            comb_list.append("G+" + str(i) + "G-" + str(j))
        
    return area_list,place_list,comb_list

if __name__ == "__main__":
    if flip == True:
        areas = []
        comb = []
        place = []
        for i in range(8):
            tmp = ["A"]
            tmp_i = i
            for j in range(3):
                check = tmp_i % 2
                tmp_i = int(tmp_i / 2)
                if check == 0:
                    if j == 0:
                         tmp.append("B")
                    if j == 1:
                         tmp.append("C")
                    if j == 2:
                         tmp.append("D")
                else:
                    if j == 0:
                         tmp.append("B_")
                    if j == 1:
                         tmp.append("C_")
                    if j == 2:
                         tmp.append("D_")
            rect = trans(tmp)
            array = make_parm(tmp)
            area_list,place_list,comb_list = main(array,rect)        
            areas.extend(area_list)
            comb.extend(comb_list)
            place.extend(place_list)
        areas = np.array(areas)
        print("=============Final Result=============")

        print(np.argmin(areas))
        print(comb[np.argmin(areas)])
        print(place[np.argmin(areas)])
        print(areas[np.argmin(areas)])
    
    else:
        array = make_parm(name)
        rect = [A,B,C,D]
        area_list,place_list,comb_list = main(array,rect)
        print("=============Final Result=============")
        print(np.argmin(area_list))
        print(place_list[np.argmin(area_list)])
        print(comb_list[np.argmin(area_list)])
        print(area_list[np.argmin(area_list)])