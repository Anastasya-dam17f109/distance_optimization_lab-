import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
import random
import copy
import json
import scipy.spatial.distance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with open("task.json", 'r') as task_file:
    task = json.load(task_file)

n = task["points_number"]
# радиус сферы
rad = task["R"]
# число генерируемых точек
N = n + 3
# заданная точность
eps = task["precision"]
# максимальное число итераций
n_iter_max = task["iterations_max"]

# вычисление евклидового расстояния между оптимизируемыми точками

def euclid_dist_between(vec):
    return scipy.spatial.distance.pdist(vec, 'euclidean')

# первая оптимизируемая функция

def F_1(vec):
     R = (euclid_dist_between(to_dec(vec)))**-2
     return(R.sum())

# вторая оптимизируемая функция

def F_2(vec):
     arr = euclid_dist_between(to_dec(vec))
     return (-1.0*np.min(arr))

# преобразование координат точек из сферической системы координат в декартовую

def to_dec(vec_sp):
    vec_dec = np.zeros((vec_sp.shape[0], 3), dtype=float)
    phi = vec_sp[:, 0]
    theta = vec_sp[:, 1]
    vec_dec[:, 0] = np.cos(phi) * np.sin(theta)
    vec_dec[:, 1] = np.sin(theta) * np.sin(phi)
    vec_dec[:, 2] = np.cos(theta)
    return (vec_dec)

# задание начального положения точек на поверхности сферы

def points_generator(npoints):
     theta = 2*np.pi*np.random.rand(npoints)
     phi = np.arccos(2*np.random.rand(npoints)-1)
     vec_sp = []
     for i, j in zip(phi, theta):
         vec_sp.append([i, j])
     vec_sp = np.array(vec_sp)
     vec_dec = to_dec(vec_sp)
     return [vec_sp,vec_dec]

# одномерная оптимизация(по выбранному напрвлению) методом золотого сечения

def golden_ratio_opt(points, pt_num, axis_num, f, l_bound, r_bound):
     coeff = (1 + math.sqrt(5)) / 2
     points_buf = points.copy()

     while (( r_bound - l_bound) > eps):
        x_left = r_bound - (r_bound-l_bound) / coeff
        x_right = l_bound + (r_bound-l_bound) / coeff

        points_buf[pt_num][axis_num] = x_left
        f_at_left = f(points_buf)

        points_buf[pt_num][axis_num] = x_right
        f_at_right = f(points_buf)
        if (f_at_left >= f_at_right):
            l_bound = x_left
        else:
            r_bound = x_right

     res = (l_bound+ r_bound) / 2
     points[pt_num][axis_num] = res

# многомерная оптимизация методом Гаусса-Зейделя

def gauss_zeidel_opt(points, N, func):
     xN_1 = points.copy()
     xN = points.copy()
     is_optimized = False
     ctr = 0
     while not is_optimized and (ctr < n_iter_max ):
         for i in range(0, N):
             golden_ratio_opt(xN, i, 0, func, 0, math.pi * 2)
             golden_ratio_opt(xN, i, 1, func, 0, math.pi)
         dist = np.linalg.norm(to_dec(xN_1)-to_dec(xN))
         if (eps >= dist):
            is_optimized = True
         xN_1 = xN.copy()
         ctr += 1
     return xN


print(" генерация оптимизируемых точек ")
vec_sp1, np_points_start = points_generator(N)
vec_sp2 =  vec_sp1.copy()

print(" оптимизация по методу Гаусса-Зейделя первой функциии ")
points1 = gauss_zeidel_opt(vec_sp1, N, F_1)
np_points1 = np.array(to_dec(points1))

print(" оптимизация по методу Гаусса-Зейделя второй функции ")
points2 = gauss_zeidel_opt(vec_sp2, N, F_2)
np_points2 = np.array(to_dec(points2))

# оттрисовка графики - сферической поверхности и точек на ней
fig =make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['first function', 'second function'] )

# оптимизированные точки
fig.add_trace(
    go.Scatter3d(
         x=np_points1[:,0],y=np_points1[:,1],z=np_points1[:,2], mode='markers',name='end position',
    marker_color='rgba(255, 0, 0, .5)'),  1, 1)
fig.add_trace(
    go.Scatter3d(
         x=np_points2[:,0],y=np_points2[:,1],z=np_points2[:,2], mode='markers',name='end position',
    marker_color='rgba(255, 0, 0, .5)'),  1, 2)

# начальное положение точек на поверхности
fig.add_trace(
    go.Scatter3d(
    x=np_points_start[:,0],y=np_points_start[:,1],z=np_points_start[:,2],
    mode='markers',name='initial position',
    marker_color='rgba(255, 182, 0, .5)'),  1, 1)
fig.add_trace(
    go.Scatter3d(
    x=np_points_start[:,0],y=np_points_start[:,1],z=np_points_start[:,2],
    mode='markers',name='initial position',
    marker_color='rgba(255, 182, 0, .5)'),  1, 2)

# построение (по точкам)  поверхности сферы радиуса 1
phi, theta  = np.mgrid[0:2*np.pi:100j, 0:2*np.pi:100j]
x0 = rad * np.cos(theta)*np.sin(phi)
y0 = rad * np.sin(theta)*np.sin(phi)
z0 = rad * np.cos(phi)
sphere = go.Surface(x=x0, y=y0, z=z0, colorscale='Blues',opacity=0.3,showscale=False)
fig.add_trace(sphere, 1, 1)
fig.add_trace(sphere, 1, 2)
fig.show()
