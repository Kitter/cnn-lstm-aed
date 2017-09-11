import math
from scipy import stats

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2GRAY
from sys import float_info




def int2round(src):
    """
    returns rounded integer recursively
    :param src:
    :return:
    """
    if isinstance(src, float):
        return int(round(src))

    elif isinstance(src, tuple):
        res = []
        for i in range(len(src)):
            res.append(int(round(src[i])))
        return tuple(res)

    elif isinstance(src, list):
        res = []
        for i in range(len(src)):
            res.append(int2round(src[i]))
        return res
    elif isinstance(src, int):
        return src
    if isinstance(src, str):
        return int(src)


def rescale_domain(a, range_):
    aa = a.reshape(a.size)
    oldMin = min(aa)
    oldMax = max(aa)
    newMin = range_[0]
    newMax = range_[1]

    OldRange = (oldMax - oldMin)
    NewRange = (newMax - newMin)

    b = []
    for e in aa:
        n = (((e - oldMin) * NewRange) / OldRange) + newMin
        b.append(n)
    bb = np.array(b)

    return bb.reshape(a.shape)


def sorting_by_col(_A, col):
    """

    :param A: given n by m matrix
    :param col: sorting based on selected column.
    :return:
    """
    A = np.array(_A)
    return A[A[:, col].argsort()]





def plot_Graph(domains, codomains, row, title=None, path=None, fname=None, isAxis=True, isDot=True, labels=None,
               subtitle=None, legend_loc=4, axis_label=None):
    """
     x1, y1,y2,
     [ X1 ], [  [Y1 Y2 Y3] ] plot Y1~Y3 in single graph.
     [X1, X2], [ [Y1] [ Y2] ] plot X1-Y1 and X2-Y2 in two graphs.
    :param domains: [ x1, x2 , x3 ]
    :param codomains: [ [y1,y2,y3] , [ y1,y2]  ]
    :param labels:  [ t1 ,t2 ,t3 ,t4]
    :param row:
    :param path:
    :param fname:
    :param axis_label: axis_labels for each graph [ [x_label, y_label] [] [] ]
    :return:
    """

    # assert (len(domains) == len(codomains)), 'different dimension between domain and codomain'
    col = len(codomains) // row
    if col * row < len(codomains):
        col += 1

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    if isDot:
        plot_type = ['bo', 'go', 'ro', 'co', 'mo', 'yo']
    else:
        plot_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-']

    if isAxis:
        axis_flag = 'on'
    else:
        axis_flag = 'off'

    plt.figure(1)
    if title:
        plt.suptitle(title)

    for n in range(len(codomains)):
        legends = []
        print(col, row, n)
        ax = plt.subplot(col, row, n + 1)

        if subtitle:
            plt.title(subtitle[n])

        plt.axis(axis_flag)

        if axis_label:
            plt.xlabel(axis_label[n][0])
            plt.ylabel(axis_label[n][1])

        for k in range(len(codomains[n])):

            if len(domains) == 1:
                # multi-codomain for single domain.
                ax.plot(domains[0], codomains[n][k], plot_type[k % len(colors)], label=labels[k])

            else:
                # each codomain has pair domain.
                if len(codomains[n]) == 1:
                    # one domain for one codomain
                    print('n: ', n, ' k: ', k)
                    ax.plot(domains[n], codomains[n][k], plot_type[n % len(colors)], label=labels[n])

                else:
                    # one domain for multi-codomain
                    ax.plot(domains[n], codomains[n][k], plot_type[k % len(colors)], label=labels[k])

        ax.legend(loc='best')
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '/' + fname + '.png')

    plt.clf()

    plt.close()


def polar_coordiates(pt1, pt2, isPoint):
    """

    :param pt1:
    :param pt2:
    :return: degree value
    """
    if isPoint:
        x = pt2[0] - pt1[0]
        y = pt2[1] - pt1[1]
    else:
        x = pt1
        y = pt2

    pi = np.degrees(np.pi)
    if x > 0 and y >= 0:
        return np.degrees(np.arctan(y / float(x)))
    elif x > 0 and y < 0:
        return np.degrees(np.arctan(y / float(x))) + (pi * 2)
    elif x < 0:
        return np.degrees(np.arctan(y / float(x))) + pi
    elif x == 0 and y > 0:
        return pi / 2
    elif x == 0 and y < 0:
        return pi * 1.5
    else:
        return -1


def gradient(pt1, pt2):
    delta_x = pt1[0] - pt2[0]
    delta_y = pt1[1] - pt2[1]

    m = delta_y / float(delta_x)

    return m

def euclidean_dist(pt1,pt2):
    delta_x = pt1[0] - pt2[0]
    delta_y = pt1[1] - pt2[1]
    dist = np.square(delta_x)+ np.square(delta_y)
    return np.sqrt(dist)



def split_into(size, step):
    n = size / step
    q = size % step
    ranges = []
    print(size, n, q)
    for i in range(n):
        ranges.append((i * step, (i + 1) * step))
    ranges.append((n * step, (n * step) + q))

    print(ranges)






def sobel(a, axis):
    """
    :param a: given matrix, gray-sacle
    :param axis:  0 for x-axis, 1 for y-axis
    :return:
    """

    assert (len(a.shape) == 2), 'given matrix should be gray-scale.'

    d = np.zeros(a.shape)
    kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # default x-axis

    if axis == 0:
        kx = np.array(kx)
    else:
        kx = np.array(kx).T

    for y in range(1, a.shape[0] - 1):
        for x in range(1, a.shape[1] - 1):
            aa = a[y - 1:y + 2, x - 1:x + 2]
            d[y, x] = sum(sum(aa * kx))

    return d


def center_to_upper(rect):
    """

    :param rect: center(x,y) and (w,h)
    :return: [x1, x2, y1 ,y2 ]
    """
    xc = rect[0]
    yc = rect[1]
    w = rect[2]
    h = rect[3]

    ux = xc - (w / 2)
    uy = yc - (h / 2)
    return (ux, ux + w, uy, uy + h)


def overlapping_area_size(a, b):
    """

    :param a: (x1, x2, y1, y2)
    :param b: (x1, x2, y1, y2)
    :return:
    """
    dx = min(a[1], b[1]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[2], b[2])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return -1


def progress_bar(i, l, m=20):

    indicator = divmod(l,m)[0]
    a,b = divmod(i,indicator)
    d = int(a)


    if b == 0 :
        bar = ('='*d )+'>' + '.'*(m-d)
        bar += '  '+ ' %2d %% '%(100*i/l ) + '    ' + str(i)
        print(bar)
    elif i == l-1:

        bar = ('=' * m) + '>'
        bar += '  ' + '100 %'+'    '+  'DONE'
        print(bar)



def progress_bar2(i,l,m=20):

    if i == l-1:
        print( 'Done')
    else:
        mod = divmod(l,m)[0]
        q,r = divmod(i,mod)

        if r == 0:
            d = int(q)
            bar = ('='* d)
            print()





def get_indices(total, batch):
    indices = []

    isize = divmod(total, batch)[0]
    for i in range(isize):
        indices.append(np.arange(i * batch, (i + 1) * batch))
    indices.append(np.arange(isize * batch, total))
    return indices


def ex1():
    x = []
    for i in range(2,10):
        x.append(np.ones((5)) * i)
    return np.array(x)



