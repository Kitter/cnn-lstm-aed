# PROJECT_PATH='/home/peter/PycharmProjects/xu_encoding'
PROJECT_PATH='F:\Pycharm_Projects\cnn_video_representation'
import sys,os
sys.path.append(PROJECT_PATH)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import var
from cnn.machrisaa_vgg16 import Vgg16
from ict.trainer import trainer
import cnn.utils as utils

import tensorflow as tf

from collections import namedtuple

HYPER_PARAMS = namedtuple('HYPER_PARAMS',
                          'cnn_ver,'
                          'clip_n,'
                          'vlad_k,'
                          'pca_k'
                          )

if __name__ == '__main__':
    args = sys.argv

    hps_set2 = [
        # HYPER_PARAMS(cnn_ver=16, clip_n=10, vlad_k=3, fl='f6'), # 0
        #
        # HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=3, fl='f6'), # 1
        # HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=6, fl='f6'), # 2
        #
        # HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=3, fl='nf6'), # 3
        # HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=6, fl='nf6'), # 4
        #
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=5, fl='f6'), # 5
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=9, fl='f6'), # 6
        #
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=9, fl='nf6'), # 7
        # HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=5, fl='nf6'), # 8
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=5, fl='nf6'), # 9
        #
        # HYPER_PARAMS(cnn_ver=8, clip_n=20, vlad_k=6, fl='nf6'),  # 10
        # HYPER_PARAMS(cnn_ver=8, clip_n=30, vlad_k=9, fl='nf6'),  # 11
        #
        #
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=256, fl='nf6'),  # 12
        # HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=128, fl='nf6'),  # 13
        #
        #
        # HYPER_PARAMS(cnn_ver=8, clip_n=20, vlad_k=128, fl='nf6'),  # 14
        # HYPER_PARAMS(cnn_ver=8, clip_n=20, vlad_k=64, fl='nf6'),  # 15
        #
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=128, fl='nf6'),  # 16
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=64, fl='nf6'),  # 17
        # HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=64, fl='f6'),  # 18

        HYPER_PARAMS(cnn_ver=16, clip_n=15, vlad_k=64, pca_k=512),  # 0
        HYPER_PARAMS(cnn_ver=16, clip_n=15, vlad_k=64, pca_k=256),  # 1
        HYPER_PARAMS(cnn_ver=16, clip_n=15, vlad_k=64, pca_k=128),  # 2


        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=64, pca_k=512),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=32, pca_k=512),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=16, pca_k=512),  # 3


        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=64, pca_k=256),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=32, pca_k=256),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=16, pca_k=256),  # 3


        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=64, pca_k=128),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=32, pca_k=128),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=30, vlad_k=16, pca_k=128),  # 3



        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=64, pca_k=512),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=32, pca_k=512),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=16, pca_k=512),  # 3


        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=64, pca_k=256),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=32, pca_k=256),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=16, pca_k=256),  # 3


        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=64, pca_k=128),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=32, pca_k=128),  # 3
        HYPER_PARAMS(cnn_ver=16, clip_n=20, vlad_k=16, pca_k=128),  # 3


    ]



    if len(args)> 1:

        pos = int(args[1])
        hps = hps_set2[pos]
        print(hps)



    else:

        for hps in hps_set2[4:5]:

            vgg = Vgg16(False , os.path.join(var.PROJECT_PATH, 'cnn/vgg16_weights.npz'))
            # train = trainer(hps, vgg)
            import numpy as np
            import cv2

            print(hps)



            img1 = cv2.imread(var.PROJECT_PATH+"/tmp/tiger.jpeg")
            img2 = cv2.imread(var.PROJECT_PATH+"/tmp/puzzle.jpeg")
            img3 = utils.load_image(var.PROJECT_PATH+"/tmp/puzzle.jpeg")
            img4 = utils.load_image(var.PROJECT_PATH+"/tmp/tiger.jpeg")


            resized_img1= cv2.resize(img1,(224,224))
            resized_img2= cv2.resize(img2,(224,224))


            batch = np.array([resized_img1,resized_img2,img3,img4]).astype(np.float32)

            with tf.Session() as sess:
                images = tf.placeholder("float", [None, 224, 224, 3])
                feed_dict = {images: batch}

                with tf.name_scope("content_vgg"):
                    vgg.build(images)

                prob = sess.run(vgg.prob, feed_dict=  feed_dict)
                utils.print_prob(prob[0], var.PROJECT_PATH+'/cnn/sysnet.txt')
                utils.print_prob(prob[1], var.PROJECT_PATH+'/cnn/sysnet.txt')
                utils.print_prob(prob[2], var.PROJECT_PATH+'/cnn/sysnet.txt')
                utils.print_prob(prob[3], var.PROJECT_PATH+'/cnn/sysnet.txt')

            # train.make_cnn_feature22()
            # train.plotting_each_cam('fc6')
            # train.plotting('fc6')
            # train.plotting('vlad')
            # train.dimension_reduction()
            # train.train_kmeans()
            # train.encoding()
            # train.cross_validation()



        # runner().visual_real_time()



