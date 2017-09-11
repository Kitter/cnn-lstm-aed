import numpy as np
from tiah.tools import *
import pickle
from sklearn.cluster import KMeans

def test_dic():
    a = {}
    a['ba'] = (3, 3, 4)
    a['bd'] = (3, 2, 'asd')

    for k in a.keys():
        print(a[k])




def nor_test():
    from tiah import tools as tools
    from sklearn.preprocessing import normalize

    x = tools.ex1()

    # a = [ [1,4] , [ 2,7]]
    # x = np.array(a,dtype=np.float32)
    x = [[0, 3, 4], [1, 3, 4]]
    x = np.array(x, dtype=np.float32)

    print(normalize(x, axis=1))  # axis=0 is for feature

    # print(normalize(x,axis=1))


def abc():
    a = 'f:/DATASET/12. ActivityNet/activitynet/yDCZNNI3tDo.webm'

    from tiah import ImageHandler as im
    from download.read_json import get_data_dic

    vali, train, test = get_data_dic()
    import var, os, glob

    count = 0
    for vi in train:

        id = vi['url'].split('=')[1]

        infiles = glob.glob(os.path.join(var.ACTIVITY__VIDEO_PATH, id + '.*'))
        if len(infiles) == 0:
            continue
        infile = infiles[0]
        cap, prop = im.read_video_by_path(path=infile)
        if prop['fps'] > 100:
            print(prop)
            count += 1

    print('total %d outliers ' % (count))


def sum_test():
    x = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    centers = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    X = np.array(x)

    centers = np.array([1, 1, 1, 2, 2])
    label = np.array([1, 1, 1, 2, 2])
    V = np.zeros([3, 3])

    print(label == 1)
    print (X[label == 1, :])


def bar_test():

    for i in range(100):
        l = 100
        progress_bar(i,l)
from sklearn import svm

def svc_test():
    print('aaaaaaaaaaa')
    X = np.array([[0, 0], [1, 1]])
    y = ['abd', 'eee']
    clf = svm.SVC(kernel='precomputed')
    # linear kernel computation
    gram = np.dot(X, X.T)
    clf.fit(gram, y)
    clf.predict(gram)



def lda_test():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    f6 = np.load('F:/DATASET/13. GIST/1.violence/vgg16/cam1/fc6/fc6_00118.npy')
    f7 = np.load('F:/DATASET/13. GIST/1.violence/vgg16/cam1/fc7/fc7_00118.npy')

    lda = LinearDiscriminantAnalysis(n_components=512)
    X_r2 = lda.fit_transform(f6)
    print(X_r2.shape)

def mm(a,b):
    return np.mean(np.equal(a,b).astype(np.uint8))

def wha():

    clip_path = 'F:/DATASET/13. GIST/1.violence/vgg16/clip20'
    vlad_path ='F:/DATASET/13. GIST/1.violence/vgg16/clip20_vlad6'


    clip_train = clip_path+'/cam2'
    clip_test = clip_path+'/cam2ab'

    vlad_train = vlad_path+'/cam2'
    vlad_test = vlad_path+'/cam2ab'

    xname = pickle.load(open(clip_train+'/fea_nameX','rb'))
    for x in xname:
        af6 = np.load(clip_train+'/f6_'+x)
        anf6 = np.load(clip_train+'/nf6_'+x)
        avf6 = np.load(vlad_train+'/vlad_f6_'+x)
        avnf6 = np.load(vlad_train+'/vlad_nf6_'+x)

        bf6 = np.load(clip_test + '/f6_' + x)
        bnf6 = np.load(clip_test + '/nf6_' + x)
        bvf6 = np.load(vlad_test + '/vlad_f6_' + x)
        bvnf6 = np.load(vlad_test + '/vlad_nf6_' + x)

        print(x,mm(af6,bf6), mm(anf6,bnf6), mm(avf6,bvf6),mm(bvnf6,avnf6))

def adf():
    from scipy.cluster.vq import kmeans2



    clip_path = 'F:/DATASET/13. GIST/1.violence/vgg16/clip20'
    clip_train = clip_path + '/cam2'
    xname = pickle.load(open(clip_train+'/fea_nameX','rb'))
    x = xname[0]
    y0 = [ 0 for x in range(20)]
    # y1 = [ 1 for x in range(10)]
    y =np.array(y0)
    xx = 'nf6_'+x
    fx = np.load(clip_train+'/'+xx)
    est = KMeans(n_clusters=4, init='k-means++', tol=0.0001)

    a = est.fit_transform(fx,y)
    b = est.fit_transform(fx,y)
    c = est.fit_transform(fx,y)

    print(np.mean(np.equal(a,b).astype(np.uint8)))
    print(np.mean(np.equal(c,b).astype(np.uint8)))
    print(np.mean(np.equal(a,c).astype(np.uint8)))

    mnit = 'points'
    c1 = kmeans2(fx ,k = 3, iter=10,thresh=1e-05,minit=mnit ,missing='warn', check_finite=True)
    c2 = kmeans2(fx ,k = 3, iter=10,thresh=1e-05,minit=mnit ,missing='warn', check_finite=True)
    c3 = kmeans2(fx ,k = 3, iter=10,thresh=1e-05,minit=mnit ,missing='warn', check_finite=True)

    c1 = c1[0].ravel()
    c2 = c2[0].ravel()
    c3 = c3[0].ravel()

    print(np.mean(np.equal(c1,c2).astype(np.uint8)))
    print(np.mean(np.equal(c2,c3).astype(np.uint8)))
    print(np.mean(np.equal(c1,c3).astype(np.uint8)))


def a3r():
    sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']

    path = 'F:/DATASET/13. GIST/1.violence/vgg16/clip30'
    for sample_nxt in sample_list:  # for each video
        clip_cam_path = path+'/'+ sample_nxt
        y = pickle.load(open(clip_cam_path + '/fea_nameY', 'rb'))
        y = np.array(y)
        c0 = len(np.where(y==0)[0])
        c1 = len(np.where(y==1)[0])
        idx = np.where(y==255)
        print(idx, np.unique(y))
        # print(sample_nxt, 'pos: ', c0, ' neg: ', c1 )


import os


from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from ict.read_txt import *
import cv2
def make_cnn_feature22():


    sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
    src_path = var.GIST_VIOLENCE_PATH

    total_fes, total_pos, total_neg = 0, 0, 0


    for sample_nxt in sample_list:  # for each video
        print(sample_nxt, end=' ', flush=True)

        bboxX = get_bbox(src_path, sample_nxt)  # bounding-box of video
        labelX = frame_label(sample_nxt)


        keys = list(bboxX.keys())
        n_frame = 13000
        cap, props = im.read_video_by_path(src_path,sample_nxt+'.avi')
        frames =[]
        i = -1

        while 1:
            ret, frame = cap.read()
            i +=1
            if ret is False:
                break

            frames.append(frame)

            if var.DEBUG:
                tools.progress_bar(i, n_frame, 10)

            if (i in keys):
                bbox = bboxX[i]

                for k in bbox.keys():
                    x1, x2, y1, y2 = bbox[k]
                    height = y2-y1
                    height = height * 1.1
                    height2 = int(height/2)
                    cx = int( (x1+x2)/2)

                    x1_ = cx- height2
                    x2_ = cx+ height2

                    cv2.rectangle(frame,(x1_,y1),(x2_,y2),(0,0,255),2)
                    cv2.imshow('1',frame)
                    cv2.waitKey(10)



def aa3():
    path = var.PROJECT_PATH+'/cnn'
    w1_file = path+'/vgg16_weights.npz'
    w2_file = path+'/vgg16_weights2.npz'

    w1 = np.load(w1_file)
    w2 = np.load(w2_file)
    keys = sorted(w1)
    for i ,k in enumerate(keys):

        a = np.mean(np.equal(w1[k],w2[k]).astype(np.uint8))
        print(i, k, a)

aa3()