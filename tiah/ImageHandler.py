from os import listdir

import cv2
import numpy as np
from natsort import natsorted
from scipy import stats

from tiah import tools


def split_video(src, video_props, start, end, path, fname):
    fps = video_props[1]
    res = src[start * fps:end * fps]
    write_video(res, video_props, path, fname)


def write_video(src, fps, path, fname, p=None):
    """
    :param src:
    :param video_props: fourcc , fps , size
    :param path:
    :param fname:
    :return:
    """

    print ('writing vidoe file ', len(src))
    if src is None:
        print ('tools , write video : no list')
        return None
    elif len(src) == 0:
        print ('tools , write video : empty list')
        return None
    else:
        size = src[0].shape
        if p is None:
            fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
            writer = cv2.VideoWriter(path + '/' + fname + '.avi', fourcc, fps, (size[1], size[0]))
        else:
            writer = cv2.VideoWriter(path + '/' + fname + '.avi', int(p['fourcc']), p['fps'], (size[1], size[0]))

        if  len(size) == 2:
            for img in src:
                writer.write(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
        else:
            for img in src:
                writer.write(img)

        writer.release()


def write_imgs(src, path, fname):
    if isinstance(src, list) or type(src)==type(np.array([])):
        n = len(str(len(src)))
        for i in range(len(src)):
            cv2.imwrite(path + "/" + fname + str(i).zfill(n) + '.png', src[i])
    else:
        cv2.imwrite(path + "/" + fname+'.jpg', src)

def read_file_list(path,sort=True):
    flist = listdir(path)
    if sort :
        return natsorted(flist)
    else:
        return flist


def read_images_by_path(path , flag_color = True, isnumpy=False):
    print ('reading images from ',path , flag_color)
    file_list = listdir(path)
    file_list = natsorted(file_list)
    img_list = []

    if flag_color:
        for file in file_list:
            img_list.append(cv2.imread(path+'/'+file , flags=1)) # flag=1 : color
    else:
        for file in file_list:
            img_list.append(cv2.imread(path + '/' + file,  flags=0)) # flag=2 : grayscale
    if isnumpy:
        return np.array(img_list)
    else:
        return img_list

def show_video(path, fname, fps):
    """

    :param path: path to video file
    :param fname: video file name
    :return: image list
    """
    print("video reading  from " , path + '/' + fname)
    cap = cv2.VideoCapture()
    cap.open(path + '/' + fname)


    if not cap.isOpened():
        print ('tools, read video as list by path : file not exist')
        return None

    count = 0
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break

        cv2.putText(frame, 'Index: %d '% (count), (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3,
                    (0, 0, 0), thickness=2)
        cv2.imshow('1,',frame)
        if cv2.waitKey(fps) & 0xff == 113:  # 113 refers to 'q'
            break

        count +=1



    cv2.destroyAllWindows()
    cap.release()



def read_video_as_list_by_path(path, fname_ext = None ,color_flag=1, print_flag= 0):
    """

    :param path: path to video file
    :param fname: video file name
    :return: image list
    """

    cap = cv2.VideoCapture()
    if fname_ext is None:
        cap.open(path)
    else:
        cap.open(path + '/' + fname_ext)


    if not cap.isOpened():
        if fname_ext is None:
            print('not exist ', path)
        else:
            print('not exist ', path+'/'+fname_ext)
        quit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    props = { 'fps':int(fps), 'fourcc': int(fourcc), 'width': int(width), 'height': int(height), 'length':int(length)}
    f = []
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break
        if color_flag:
            f.append(frame)
        else:
            f.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if len(f) is 0:
        print('tools, read video as list by path : no frames in video')
        quit()

    cap.release()

    if print_flag:
        print('Video size: ', len(f), ' params: ' ,(fps,width,height) )

    return f, props

def read_video_by_path(path , fname = None , isprint=0):
    """

    :param path: path to video file
    :param fname: video file name
    :return: image list
    """

    cap = cv2.VideoCapture()

    if fname is None:
        # print("video reading from " , path)
        cap.open(path)
    else:
        # print("video reading from " , path + '/' + fname)
        cap.open(path + '/' + fname)





    if not cap.isOpened():
        print ('tools,', path + '/' + fname,  'file not exist')
        quit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    props = { 'fps':int(fps), 'fourcc': int(fourcc), 'width': int(width), 'height': int(height), 'length':int(length)}
    if isprint:
        print(props)
    return cap , props


def get_list_draw_every_ellipse(src, ellipse):
    """
    drawing ellipses for each image.

    :param src: image list
    :param ellipse: ellipse(center,size,orientation) list
    :return: image list
    """
    show_list = []

    idx = 0
    for i in range(len(src)):
        img = src[i].copy()
        if len(ellipse[i]) != 0:

            for j in range(len(ellipse[i])):  # draw all contours and line corresponding to contour.
                cv2.ellipse(img, ellipse[i][j], (0, 255, 0), 1)
                ellipse_center = tools.int2round(ellipse[i][j][0])
                cv2.putText(img, str(idx) , ellipse_center,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255,255,255))
                idx +=1
            show_list.append(img)

    return show_list


def display_img(input_list, waitSec=30):
    for img in input_list:
        cv2.imshow('1', img)
        if cv2.waitKey(waitSec) & 0xff == 113:  # 113 refers to 'q'
            break
    cv2.destroyAllWindows()


def get_list_fgmask(src, history=30, nmixtures=10, ratio=0.1):
    """
    background subtraction.
    returns binary image
    """
    print ('masking foreground')
    fgbg = cv2.BackgroundSubtractorMOG(history=history, nmixtures=nmixtures, backgroundRatio=ratio)
    fgmask_list = []
    dp_list = []
    for frame in src:
        fgmask = fgbg.apply(frame)
        dp_list.append(fgmask)
        fgmask_list.append(stats.threshold(fgmask, threshmin=0,threshmax=0, newval=1))

    return fgmask_list,dp_list

def resize(src, shape):
    resized = []
    for img in src:
        resized.append(cv2.resize(img,shape))
    return resized

def to_gray(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)