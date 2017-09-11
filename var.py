
import os
from cv2 import FONT_HERSHEY_COMPLEX
import tiah.FileManager as fm
from os import path
# PROJECT_PATH = 'F:/Pycharm_Projects/cnn_video_representation'
# PROJECT_PATH = '/home/peter/PycharmProjects/xu_encoding'
PROJECT_PATH='F:\Pycharm_Projects\cnn_video_representation2'

ACTIVITY_PATH = os.path.join('f:/DATASET/12. ActivityNet')
ACTIVITY__VIDEO_PATH = os.path.join(ACTIVITY_PATH, 'activitynet')

GIST_VIOLENCE_PATH = os.path.join('f:/DATASET/13. GIST/1.violence')
# GIST_VIOLENCE_PATH = os.path.join('/home/peter/datasets/gist/1.violence')

GIST_KIDNAP_PATH = os.path.join('f:/DATASET/13. GIST/2.kidnap')
GIST_ROBBERY_PATH = os.path.join('f:/DATASET/13. GIST/3.robbery')

# TRAIN_X_PATH = os.path.join(ACTIVITY_PATH,'vgg16/feature_train')
# TEST_X_PATH = os.path.join(ACTIVITY_PATH,'vgg16/feature_test')
# VALI_X_PATH = os.path.join(ACTIVITY_PATH,'vgg16/feature_vali')



SIG_WEIGHT = '_W'
SIG_BIAS = '_b'
FILTER = 'filter'
BIAS = 'biases'
WEIGHT = 'weights'




# TRAIN_FC6 = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_train'))
# TRAIN_FC6_VLAD = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_train_vlad'))
# TRAIN_FC6_PCA = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_train_pca'))
# TRAIN_FC6_FV = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_train_fv'))


# VALID_FC6 = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_valid'))
# VALID_FC6_VLAD = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_valid_vlad'))
# VALID_FC6_PCA = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_valid_pca'))
# VALID_FC6_FV = fm.mkdir(path.join(ACTIVITY_PATH,'fc6_valid_fv'))


KEYS = 'keysK'
LABEL = 'labelY'
FEATURE ='featureX'

WEIGHT_PATH = os.path.join(PROJECT_PATH, 'cnn/vgg16_weights.npz')
TUNED_WEIGHT_PATH = os.path.join(PROJECT_PATH, 'cnn/tuned_vgg16_weights.npz')

white = (255, 255, 255)
red = (0, 0, 255)
green = (0, 255, 0)
black = (0,0,0)
FONT_FACE = FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.8
thickness = 2
# DEBUG = True
DEBUG = True