
# by Grossmend
# January, 2018

import sys
import random as rnd
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.spatial import distance


# функция генерации координат x и y, с учетом кластеров
def arrdata(fn, fcluster_count):

    if fn <= fcluster_count or fcluster_count <= 0:
        text_error = "function 'arrdata': the number of points can not be less than"
        text_error += " or equal number of clusters or number cluster <= 0, exit"
        sys.exit(text_error)

    x_co = np.array(())
    y_co = np.array(())

    for cls in range(1, fcluster_count + 1):
        sctr = int(rnd.random() * 5)
        sign = round(rnd.uniform(0, 3))
        for i in range(1, round(fn / fcluster_count)+1):
            if sign == 0:
                x_co = np.append(x_co, round((rnd.normalvariate(0, 1) + sctr) * 100))
                y_co = np.append(y_co, round((rnd.normalvariate(0, 1) + sctr) * 100))
            elif sign == 1:
                x_co = np.append(x_co, round((rnd.normalvariate(0, 1) + sctr) * 100))
                y_co = np.append(y_co, round((rnd.normalvariate(0, 1) - sctr) * 100))
            elif sign == 2:
                x_co = np.append(x_co, round((rnd.normalvariate(0, 1) - sctr) * 100))
                y_co = np.append(y_co, round((rnd.normalvariate(0, 1) + sctr) * 100))
            elif sign == 3:
                x_co = np.append(x_co, round((rnd.normalvariate(0, 1) - sctr) * 100))
                y_co = np.append(y_co, round((rnd.normalvariate(0, 1) - sctr) * 100))

    return x_co, y_co


# функция вычисления матрицы дистанций
def distcalc(x_cord, y_cord):
    # calculate numpy
    arr = np.column_stack((x_cord, y_cord))
    return distance.cdist(arr, arr, 'euclidean')


# главная функция k-medoids
def k_medoids(fdist, k):
    fn = np.size(fdist, 1)
    center = np.array(rnd.sample(range(0, fn-1), k))
    count = 0
    cost_past = 0
    while 1:
        fdistrib = clustering(fdist, center)
        center, cost_present = new_center(fdist, fdistrib, k)
        if count > 0 and cost_present == cost_past:
            break
        cost_past = cost_present
        count += 1
    return fdistrib


# функция разбрасывания точек на кластеры
def clustering(cl_dist, cl_center):
    cl_s = cl_dist[cl_center, :]
    cl_distrib = np.argmin(cl_s, axis=0)
    return cl_distrib


# функция вычисления новых медоидов
def new_center(new_dist, new_distrib, new_k):
    cost = np.zeros(new_k, dtype=int)
    # cost[:] = 0
    center = np.zeros(new_k, dtype=int)
    for i in range(0, new_k):
        # находим индексы i-ого кластера
        distrib_cluster = np.nonzero(new_distrib == i)
        temp = np.sum(new_dist[np.ix_(distrib_cluster[0], distrib_cluster[0])], axis=1)
        min_idx = np.argmin(temp, axis=0)
        cost[i] = temp[min_idx]
        center[i] = distrib_cluster[0][min_idx]
    cost = sum(cost)
    return center, cost


# функция создания массива цветов
def colors_array(len_arr):
    fcolormap = ['red', 'green', 'blue', 'grey', 'black', 'sienna', 'purple']
    fcolormap += ['sandybrown', 'deepskyblue', 'gold', 'darkcyan', 'chartreuse', 'silver']
    if len_arr > len(fcolormap):
        sys.exit("function 'colors_array': Длина входного параметра != массиву цветов")
    ans = np.array(fcolormap[0:len_arr])
    return ans


start_time = time.time()    # засекаем время работы программы

# параметры:
n = 1000   # кол-во точек
cluster_count = 3   # кол-во кластеров
gr = 1  # view

# главный код:
x, y = arrdata(n, cluster_count)    # создаем координаты x и y
dist = distcalc(x, y)   # получаем матрицу дистанций
distrib = k_medoids(dist, cluster_count)    # получаем кластеризацию (разброс точек на кластеры)

# графика:
if gr == 1:
    # преобразование координат x, y в один массив (для визуализации: так легче)
    points = np.vstack((x, y))
    # получаем массив цветов
    colormap = colors_array(cluster_count)
    # разобрать в интернете
    plt.scatter(points[0], points[1], s=5, c=colormap[distrib])
    # именованный объект
    fig = plt.gcf()
    # присвоение имени
    fig.canvas.set_window_title('my test figure')
    # наименование окна
    plt.title('clustering')
    # подпись оси x
    plt.xlabel('x point')
    # подпись оси y
    plt.ylabel('y point')
    # отображаем сетку
    # plt.grid()

print('Время работы алгоритма: ' + str(time.time() - start_time))   # конец отображения времени

# отображение графики:
plt.show()

# print(distrib)
# print('len ' + str(len(dist)))
